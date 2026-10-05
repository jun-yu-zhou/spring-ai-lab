package com.example.springailab.config;

import com.example.springailab.entity.ActorFilms;
import lombok.RequiredArgsConstructor;
import org.junit.jupiter.api.AutoClose;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.ParameterizedTypeReference;

import java.util.List;

@SpringBootTest
class AiConfigTest {

    @Autowired
    private ChatClient chatClient;

    @Test
    public void StructuredOutputTest() {
        List<ActorFilms> actorFilms = chatClient.prompt()
                .user("给我华语电影导演的前三人和他们每人各自三部代表作")
                .call()
                //validateSchema() — 根据实体架构验证 JSON 响应，并在失败时自动重试并提供错误反馈。
                //useProviderStructuredOutput() — 将架构作为 API 级约束而不是提示文本发送给提供商。
                .entity(new ParameterizedTypeReference<List<ActorFilms>>() {},
                        spec -> spec.validateSchema());
        System.out.println(actorFilms);
    }
}