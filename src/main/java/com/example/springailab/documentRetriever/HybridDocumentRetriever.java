package com.example.springailab.documentRetriever;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.redis.RedisVectorStore;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * 混合文档检索器：把「向量语义检索」和「BM25 关键词检索」两条独立通道的结果，
 * 用 RRF（Reciprocal Rank Fusion，倒数排名融合）算法合并成一个最终排名。
 *
 * <h3>为什么要两条腿？</h3>
 * <ul>
 *   <li><b>向量腿</b>：把问题交给嵌入模型转成高维向量，再到 Redis 里做 KNN 相似度匹配。
 *       它擅长"意思相近"——问"悉达多在河边领悟到了什么"，也能命中写着
 *       "智慧不可言说，只能被活出来"的段落（不依赖字面重合）；
 *       弱点是对专有名词、罕见词不敏感，且受嵌入模型质量影响。</li>
 *   <li><b>BM25 腿</b>：经典的"词频 × 逆文档频率"关键词打分算法，
 *       由 RediSearch 中文全文索引（content 字段 + friso 分词）提供。
 *       它擅长"字面精确命中"——问题里出现的关键词在文档里原样出现的段落得分高；
 *       弱点是完全不理解语义，换个说法就找不到。
 *       由于官方 searchByText 不支持设置分词语言（中文必须 LANGUAGE chinese），
 *       这条腿的具体查询由构造器注入的 bm25Leg 函数执行，
 *       实现在 RedisVectorStoreConfig#bm25Search。</li>
 * </ul>
 * 两者互补：一条腿的盲区恰好是另一条腿的强项，融合后的召回质量通常优于任何单腿。
 *
 * <h3>RRF 融合是怎么算的？</h3>
 * 对每个文档：score = Σ 1/(K + rank)，其中 rank 是该文档在某条腿结果里的名次（从 1 开始），
 * K 是平滑常数，业界惯例 60。举例（K=60）：
 * <pre>
 *   文档 A：向量腿第 1 名 → 1/(60+1) ≈ 0.0164
 *           BM25 腿第 2 名 → 1/(60+2) ≈ 0.0161   两腿都命中，分数相加 ≈ 0.0325
 *   文档 B：只在向量腿第 3 名 → 1/(60+3) ≈ 0.0159
 *   ⇒ A 的总分高于 B，尽管 A 在向量腿里同样排第 1 的对手只有一个。
 * </pre>
 * K 的作用是把"名次的细微差别"压平：由于 1/(60+rank) 对 rank 变化不敏感，
 * 任何单腿的排序噪声都不会主导最终结果，而"多腿同时命中"这一信号会被放大。
 *
 * <h3>在 RAG 流程中的位置</h3>
 * RetrievalAugmentationAdvisor 在检索阶段会调用本类的 {@link #retrieve(Query)}，
 * 把用户问题（含上下文）包装成 {@link Query} 传入；本方法返回的文档就是
 * 最终拼进提示词的知识片段。换用本类后，advisor 原有的 VectorStoreDocumentRetriever 不再使用。
 */
public class HybridDocumentRetriever implements DocumentRetriever {

    /** RRF 平滑常数。越大，名次差距对分数的影响越小（越"民主"）；60 是论文与业界最常用的取值。 */
    private static final int RRF_K = 60;

    private final RedisVectorStore vectorStore;

    /**
     * BM25 关键词腿的查询函数：(查询词, 召回数) → 命中文档列表。
     * 以函数注入而非写死调用，是因为中文场景必须用自定义的
     * FT.SEARCH + LANGUAGE chinese（官方 searchByText 不带语言参数），
     * 这样融合逻辑就不与 RediSearch 的查询细节耦合。
     */
    private final BiFunction<String, Integer, List<Document>> bm25Leg;

    /** 每条腿各自召回的候选数量。取值要比最终 topK 大，融合池才有"筛"的余地。 */
    private final int candidatesPerLeg;

    /** 融合后最终返回的文档数量。 */
    private final int topK;

    public HybridDocumentRetriever(RedisVectorStore vectorStore,
                                   BiFunction<String, Integer, List<Document>> bm25Leg,
                                   int candidatesPerLeg, int topK) {
        this.vectorStore = vectorStore;
        this.bm25Leg = bm25Leg;
        this.candidatesPerLeg = candidatesPerLeg;
        this.topK = topK;
    }

    /**
     * 检索主流程：先让两条腿各自独立跑一遍，再做 RRF 融合。
     *
     * @param query advisor 框架包装好的查询对象，query.text() 是用户问题的原文
     */
    @Override
    public List<Document> retrieve(Query query) {
        String q = query.text();

        // ---- 向量腿：问题文本 → DashScope 向量化 → Redis KNN，返回语义上最相近的 N 个 chunk ----
        List<Document> vectorHits = vectorStore.similaritySearch(
                SearchRequest.builder().query(q).topK(candidatesPerLeg).build());

        // ---- BM25 腿：问题文本 → 中文全文索引（FT.SEARCH + LANGUAGE chinese）按 BM25 打分 ----
        // 具体查询由注入的函数执行（实现在 RedisVectorStoreConfig#bm25Search）
        List<Document> textHits = bm25Leg.apply(q, candidatesPerLeg);

        // rrf:  文档 id → 累计融合分数
        // pool: 文档 id → 文档本体（去重后的候选池）
        // LinkedHashMap 保持插入顺序，便于调试时观察两条腿的先后贡献
        Map<String, Double> rrf = new LinkedHashMap<>();
        Map<String, Document> pool = new LinkedHashMap<>();
        addLeg(vectorHits, rrf, pool);
        addLeg(textHits, rrf, pool);

        // 按融合分数从高到低排序，取前 topK，再从候选池里取回文档本体
        return rrf.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(topK)
                .map(e -> pool.get(e.getKey()))
                .toList();
    }

    /**
     * 把一条腿的排名结果折算成 RRF 分数并累进总表。
     *
     * @param hits 该腿按相关性从高到低排好的文档
     * @param rrf  全局分数表（跨两条腿累加）
     * @param pool 全局候选池：putIfAbsent 保证同一文档只保留一份
     *             （两条腿都命中同一文档时，第二次出现只会累加分数，不会产生重复条目）
     */
    private void addLeg(List<Document> hits, Map<String, Double> rrf, Map<String, Document> pool) {
        for (int i = 0; i < hits.size(); i++) {
            Document d = hits.get(i);
            pool.putIfAbsent(d.getId(), d);
            // 名次 i 从 0 计，换算成从 1 开始的 rank：1/(K + rank)
            // merge：文档已在表中则累加（两腿都命中的情况），否则放入新值
            rrf.merge(d.getId(), 1.0 / (RRF_K + i + 1), Double::sum);
        }
    }
}
