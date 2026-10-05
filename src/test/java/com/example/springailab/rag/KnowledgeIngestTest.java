package com.example.springailab.rag;

import com.example.springailab.service.KnowledgeService;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;

import java.io.IOException;
import java.util.List;

@SpringBootTest
class KnowledgeIngestTest {

    @Autowired
    KnowledgeService knowledgeService;
    @Autowired
    VectorStore vectorStore;

    @Test
    void ingestAndSearch() throws IOException {
        // 1. 扫描 knowledge 目录下全部 PDF
        Resource[] pdfs = new PathMatchingResourcePatternResolver()
                .getResources("classpath:knowledge/*.pdf");

        for (Resource pdf : pdfs) {
            String fileName = pdf.getFilename();
            // 2. 入库前按文件名清理，保证测试可重复执行（否则重复跑会存两份）
            vectorStore.delete("file_name == '" + fileName + "'");
            // 3. 逐页读取，每页一个 Document（自带 file_name / page_number 元数据）
            List<Document> pages = new PagePdfDocumentReader(pdf).get();
            // 4. 切分并写入向量库
            knowledgeService.ingest(pages);
        }

        // 5. 端到端验证：向量化 → Redis 检索，应命中《悉达多》的片段
        List<Document> hits = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query("悉达多在河边领悟到了什么")
                        .topK(3)
                        .build());
        hits.forEach(d -> System.out.println(
                d.getMetadata().get("file_name") + " | " + d.getText()));
    }
}