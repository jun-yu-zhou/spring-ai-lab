package com.example.springailab.rag;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.redis.RedisVectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

@SpringBootTest
public class SearchTest {

    @Autowired
    private RedisVectorStore vectorStore;

    @Test
    public void search() {
        String query = "喧哗与骚动的作者是谁？哪一年获得诺贝尔文学奖？";
        List<Document> documents = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(query)
                        .topK(6)
                        .similarityThreshold(0.55)
                        .build()
        );
        for (Document document : documents) {
            System.out.println(document.getMetadata() + ":" + document.getText());
        }
    }
}
