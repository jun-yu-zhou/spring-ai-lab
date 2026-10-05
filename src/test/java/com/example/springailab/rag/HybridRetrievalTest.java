package com.example.springailab.rag;

import com.example.springailab.documentRetriever.HybridDocumentRetriever;
import com.example.springailab.service.KnowledgeService;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.redis.RedisVectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;

import java.io.IOException;
import java.util.List;
import java.util.function.BiFunction;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 混合检索效果对比测试。
 *
 * <p>同一个问题走三条通道，把各自命中的片段打印出来对比：
 * <ol>
 *   <li>向量腿（similaritySearch）—— 语义相似度，验证嵌入模型 + KNN 是否正常；</li>
 *   <li>BM25 腿（bm25Search）—— 关键词精确匹配，走 friso 中文分词的全文索引；
 *       若这条腿为空，先确认索引已按新结构重建（改过索引结构必须 FT.DROPINDEX），
 *       再排查查询词是否包含 RediSearch 语法字符；</li>
 *   <li>混合腿（HybridDocumentRetriever，RRF 融合）—— 应当同时包含两腿的优势命中，
 *       两腿都命中的文档排名更靠前。</li>
 * </ol>
 *
 * <p>测试自带幂等入库：先按文件名清掉旧数据再重新读取入库，可反复执行。
 */
@SpringBootTest
class HybridRetrievalTest {

    @Autowired
    KnowledgeService knowledgeService;

    // 注入具体类型 RedisVectorStore 而不是 VectorStore 接口：
    // similaritySearch 在接口上，但这里 delete 也直接用了
    @Autowired
    RedisVectorStore vectorStore;

    // BM25 腿的查询函数（RedisVectorStoreConfig 提供，走 friso 中文分词的全文索引）
    @Autowired
    BiFunction<String, Integer, List<Document>> bm25Search;

    @Test
    void hybridRetrievalComparison() throws IOException {
        // ===== 第一步：幂等入库三份书评 PDF =====
        Resource[] pdfs = new PathMatchingResourcePatternResolver()
                .getResources("classpath:knowledge/*.pdf");
        for (Resource pdf : pdfs) {
            vectorStore.delete("file_name == '" + pdf.getFilename() + "'");
            knowledgeService.ingest(new PagePdfDocumentReader(pdf).get());
        }

        // ===== 第二步：同一问题走三条通道 =====
        String query = "悉达多 河水";
        int perLeg = 5;

        // 通道一：向量腿（语义相似度）
        List<Document> vectorHits = vectorStore.similaritySearch(
                SearchRequest.builder().query(query).topK(perLeg).build());

        // 通道二：BM25 腿（自建中文全文索引，friso 分词 + BM25 打分）
        List<Document> textHits = bm25Search.apply(query, perLeg);

        // 通道三：混合（RRF 融合，每腿候选 5，最终取 5）
        List<Document> fused = new HybridDocumentRetriever(vectorStore, bm25Search, perLeg, 5)
                .retrieve(new Query(query));

        // ===== 第三步：打印对比结果 =====
        System.out.println("\n========== 查询：" + query + " ==========");
        printLeg("向量腿（语义相似度）", vectorHits);
        printLeg("BM25 腿（关键词匹配）", textHits);
        printLeg("混合（RRF 融合后）", fused);

        // ===== 第四步：断言 =====
        // 融合结果非空是底线；是否命中《悉达多》看打印输出自行判断
        assertThat(fused).isNotEmpty();
    }

    private void printLeg(String title, List<Document> hits) {
        System.out.println("\n---- " + title + "（" + hits.size() + " 条）----");
        for (int i = 0; i < hits.size(); i++) {
            Document d = hits.get(i);
            System.out.printf("%d. [%s] %s%n",
                    i + 1,
                    d.getMetadata().getOrDefault("file_name", "?"),
                    abbreviate(d.getText()));
        }
    }

    /** 把换行压成空格、截断到 60 字，方便控制台对比。 */
    private String abbreviate(String text) {
        String one = text == null ? "" : text.replaceAll("\\s+", " ").trim();
        return one.length() <= 60 ? one : one.substring(0, 60) + "…";
    }
}
