package com.example.springailab.service;

import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
public class KnowledgeService {

    private final VectorStore vectorStore;

    private final TokenTextSplitter splitter = TokenTextSplitter.builder()
            .withChunkSize(700)
            .withMinChunkSizeChars(300)
            .withMinChunkLengthToEmbed(20)
            .withKeepSeparator(true)
            .withPunctuationMarks(List.of(
                    '。', '！', '？', '；', '\n',
                    '.', '!', '?', ';'
            ))
            .build();

    public KnowledgeService(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    /**
     * 通用入库入口：调用方负责构造 Document（如 PDF 读取器的产出，自带
     * file_name / page_number 等元数据），本服务负责切分与写入向量库。
     */
    public void ingest(List<Document> documents) {
        vectorStore.write(splitter.apply(documents));
    }

    /**
     * 便捷入口：纯文本场景，手动构造元数据后复用通用入口。
     */
    public void ingest(String fileName, String fileType, String content) {
        ingest(List.of(new Document(content, Map.of(
                "file_name", fileName,
                "file_type", fileType,
                "type", "knowledge"
        ))));
    }
}