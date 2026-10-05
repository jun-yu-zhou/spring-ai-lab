package com.example.springailab.config;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.redis.RedisVectorStore;
import org.springframework.ai.vectorstore.redis.autoconfigure.RedisVectorStoreProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.jedis.JedisConnectionFactory;
import redis.clients.jedis.DefaultJedisClientConfig;
import redis.clients.jedis.RedisClient;
import redis.clients.jedis.search.FTCreateParams;
import redis.clients.jedis.search.IndexDataType;
import redis.clients.jedis.search.Query;
import redis.clients.jedis.search.SearchResult;
import redis.clients.jedis.search.schemafields.SchemaField;
import redis.clients.jedis.search.schemafields.TagField;
import redis.clients.jedis.search.schemafields.TextField;
import redis.clients.jedis.search.schemafields.VectorField;

import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * Redis 向量库 + 中文全文索引的配置中心。
 *
 * <h3>为什么建索引要手动接管？</h3>
 * Spring AI 的 RedisVectorStore 建索引时（initializeSchema=true）执行 FT.CREATE
 * 不带 LANGUAGE 参数（反编译核实），RediSearch 会用默认的英文分词器处理文本——
 * 而英文分词器会把中文字符全部跳过，导致 content 字段产生 0 个词条，
 * 之后任何 BM25 关键词查询都返回空。改为这里手动建索引并声明
 * LANGUAGE chinese（RediSearch 内置的 friso 中文分词器），BM25 对中文才有意义。
 *
 * <h3>为什么 BM25 查询也要自己写？</h3>
 * 官方的 searchByText(...) 同样没有设置查询语言的能力（反编译核实其字节码中
 * 无任何 language 处理）。RediSearch 的分词是"建索引 + 查询"两侧对称的：
 * 索引侧用 friso 切词后，查询侧也必须用同一种分词方式（setLanguage("chinese")），
 * 否则查询词和索引词对不上，照样查不到。因此 BM25 查询以
 * {@link #bm25Search(RedisClient, RedisVectorStoreProperties)} 的形式在这里提供。
 */
@Configuration
public class RedisVectorStoreConfig {

    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Jedis 客户端：复刻官方自动配置的私有构造逻辑——
     * 从 Boot 创建的 Jedis 连接工厂里取出主机/端口/密码等连接信息，
     * 包装成 RediSearch 查询所需的 RedisClient。
     * （lettuce 已从依赖排除，Boot 此时会创建 JedisConnectionFactory。）
     */
    @Bean
    public RedisClient jedisClient(JedisConnectionFactory connectionFactory) {
        return RedisClient.builder()
                .hostAndPort(connectionFactory.getHostName(), connectionFactory.getPort())
                .clientConfig(DefaultJedisClientConfig.builder()
                        .ssl(connectionFactory.isUseSsl())
                        .clientName(connectionFactory.getClientName())
                        .timeoutMillis(connectionFactory.getTimeout())
                        .password(connectionFactory.getPassword())
                        .build())
                .build();
    }

    /**
     * 向量库 bean：initializeSchema 必须为 false——
     * 建索引交给本类的 {@link #createChineseIndex}，带上中文分词配置。
     *
     * <p>注意 metadataFields 这行要保留：虽然它不再参与建索引，
     * 但运行时按元数据删除（file_name == '...'）会校验字段是否在声明列表里，
     * 它是过滤删除的"通行证"。
     */
    @Bean
    public RedisVectorStore vectorStore(RedisClient jedisClient,
                                        EmbeddingModel embeddingModel,
                                        RedisVectorStoreProperties properties) {
        RedisVectorStore store = RedisVectorStore.builder(jedisClient, embeddingModel)
                .indexName(properties.getIndexName())
                .prefix(properties.getPrefix())
                .initializeSchema(false)
                .metadataFields(RedisVectorStore.MetadataField.tag("file_name"))
                .build();

        createChineseIndex(jedisClient, embeddingModel, properties);
        return store;
    }

    /**
     * 手动创建带中文分词的全文索引。
     *
     * <p>字段与 Spring AI 写入的 JSON 文档一一对应：
     * <pre>
     * { "chunk_index":0, "page_number":1, "file_name":"悉达多书评.pdf",
     *   "content":"……正文……", "embedding":[...1024 维向量...] }
     * </pre>
     * <ul>
     *   <li>content → TEXT 字段：BM25 全文检索的主战场，索引级 language=chinese
     *       让 friso 分词器把中文句子切成词条；</li>
     *   <li>file_name → TAG 字段：支持"按文件过滤/删除"（file_name == '...'）；</li>
     *   <li>embedding → VECTOR 字段：HNSW 近似最近邻，维度从嵌入模型动态获取
     *       （text-embedding-v4 的实际维度以 API 返回为准，硬编码错了向量检索会直接失明）。</li>
     * </ul>
     * 已存在同名索引时跳过（与 Spring AI 行为一致）——
     * 副作用是改过字段结构后必须手动 FT.DROPINDEX 才会按新结构重建。
     */
    private void createChineseIndex(RedisClient client,
                                    EmbeddingModel embeddingModel,
                                    RedisVectorStoreProperties properties) {
        String indexName = properties.getIndexName();
        if (client.ftList().contains(indexName)) {
            return;
        }

        // 向量字段的算法参数：与 Spring AI 默认值保持一致。
        // 参数构造后不再修改，用 Map.of 一次性构建不可变 Map，避免多余的可变状态
        Map<String, Object> vectorAttrs = Map.of(
                "TYPE", "FLOAT32",
                "DIM", embeddingModel.dimensions(),
                "DISTANCE_METRIC", "COSINE",
                "M", 16,
                "EF_CONSTRUCTION", 200);

        FTCreateParams params = FTCreateParams.createParams()
                .on(IndexDataType.JSON)                 // 文档以 ReJSON-RL 格式存储
                .addPrefix(properties.getPrefix())      // 只索引 spring-ai:doc: 前缀的键
                .language("chinese");                   // ★ 索引侧中文分词，BM25 的生命线

        List<SchemaField> fields = List.of(
                new TextField("$.content").as("content"),
                new TagField("$.file_name").as("file_name"),
                VectorField.builder()
                        .fieldName("$.embedding").as("embedding")
                        .algorithm(VectorField.VectorAlgorithm.HNSW)
                        .attributes(vectorAttrs)
                        .build());

        client.ftCreate(indexName, params, fields);
    }

    /**
     * BM25 关键词检索，供混合检索器的关键词腿使用。
     *
     * <p>返回 BiFunction 是为了把"怎么查 BM25"抽象成一个函数——
     * HybridDocumentRetriever 只关心融合逻辑，不耦合 RediSearch 的查询细节。
     *
     * <p>查询语法 @content:(词) 表示"在 content 字段里找"；
     * setLanguage("chinese") 让查询词也经过 friso 分词，与索引侧对称。
     * 多个词之间 RediSearch 默认按 AND 处理，查不到时可用 | 改为 OR：
     * 例如 "悉达多 | 河边"。
     */
    @Bean
    public BiFunction<String, Integer, List<Document>> bm25Search(RedisClient jedisClient,
                                                                  RedisVectorStoreProperties properties) {
        return (query, limit) -> {
            Query q = new Query("@content:(" + query + ")")
                    .setLanguage("chinese")
                    .limit(0, limit);
            SearchResult result = jedisClient.ftSearch(properties.getIndexName(), q);

            List<Document> docs = new ArrayList<>();
            for (redis.clients.jedis.search.Document d : result.getDocuments()) {
                // JSON 索引的文档体在名为 "$" 的属性里，手工映射回 Spring AI 的 Document。
                // Jackson 的 readValue 抛的是受检异常，而 BiFunction.apply 不允许抛受检异常，
                // 所以这里捕获后转成非受检异常：解析失败意味着 Redis 里的数据结构变了，
                // 属于必须暴露的程序性错误，不能悄悄吞掉
                String json = d.getString("$");
                Map<String, Object> map;
                try {
                    map = objectMapper.readValue(json, new TypeReference<Map<String, Object>>() {});
                } catch (JsonProcessingException e) {
                    throw new UncheckedIOException("解析向量库文档失败: " + d.getId(), e);
                }
                docs.add(new Document(String.valueOf(map.get("content")), Map.of(
                        "file_name", String.valueOf(map.getOrDefault("file_name", "")),
                        "page_number", map.getOrDefault("page_number", 0))));
            }
            return docs;
        };
    }
}
