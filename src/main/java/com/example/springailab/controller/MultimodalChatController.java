package com.example.springailab.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.content.Media;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.util.MimeType;
import org.springframework.util.MimeTypeUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;
import reactor.core.publisher.Flux;

import java.io.IOException;
import java.net.URL;


@RestController
@RequestMapping("/multimodal")
public class MultimodalChatController {

    private final ChatClient chatClient;

    public MultimodalChatController(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    @GetMapping("/img")
    public Flux<String> stream(@RequestParam String qus,
                               @RequestParam URL imgUrl,
                               @RequestParam String sessionId) {
        return chatClient
                .prompt()
                .user(u -> u.text(qus)
                        .media(MimeTypeUtils.IMAGE_PNG, imgUrl))
                .advisors(ad -> ad.param(ChatMemory.CONVERSATION_ID, sessionId))
                .stream()
                .content();
    }

    /**
     * 浏览器直传图片：multipart 上传后只在内存中转，Spring AI 会把字节
     * 编码为 base64 data URL 随请求发给模型，不做任何落盘/存储。
     */
    @PostMapping(value = "/stream", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Flux<String> stream(@RequestParam String qus,
                               @RequestParam MultipartFile file,
                               @RequestParam String sessionId) throws IOException {
        if (file.isEmpty() || file.getContentType() == null
                || !file.getContentType().startsWith("image/")) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "仅支持上传图片文件");
        }
        // Windows 浏览器可能给出非标准的 image/jpg，规范成 image/jpeg
        String contentType = "image/jpg".equals(file.getContentType())
                ? "image/jpeg" : file.getContentType();
        Media media = Media.builder()
                .mimeType(MimeType.valueOf(contentType))
                .data(file.getBytes())
                .build();
        return chatClient
                .prompt()
                .user(u -> u.text(qus).media(media))
                .advisors(ad -> ad.param(ChatMemory.CONVERSATION_ID, sessionId))
                .stream()
                .content();
    }

}
