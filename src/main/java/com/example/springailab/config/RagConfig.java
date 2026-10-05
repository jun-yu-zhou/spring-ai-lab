package com.example.springailab.config;

import com.example.springailab.documentRetriever.HybridDocumentRetriever;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.vectorstore.redis.RedisVectorStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;
import java.util.function.BiFunction;

@Configuration
public class RagConfig {

    /**
     * RAG 检索顾问：检索阶段使用混合检索器——
     * 向量腿（语义）+ BM25 关键词腿，RRF 融合。
     *
     * <p>bm25Search 是 RedisVectorStoreConfig 提供的中文全文查询函数
     * （FT.SEARCH + LANGUAGE chinese），以 lambda 传给混合检索器的关键词腿。
     */
    @Bean
    public Advisor ragAdvisor(RedisVectorStore vectorStore,
                              BiFunction<String, Integer, List<Document>> bm25Search) {

        return RetrievalAugmentationAdvisor.builder()
                .documentRetriever(new HybridDocumentRetriever(vectorStore, bm25Search, 10, 6))
                .build();
    }
}
