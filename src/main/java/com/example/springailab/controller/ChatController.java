package com.example.springailab.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/chat")
public class ChatController {

    private final ChatClient chatClient;
    private final Advisor ragAdvisor;

    public ChatController(ChatClient chatClient,
                          Advisor ragAdvisor) {
        this.chatClient = chatClient;
        this.ragAdvisor = ragAdvisor;
    }

    @GetMapping("/ask")
    public String ask(@RequestParam String qus,
                      @RequestParam String sessionId) {
        String ans = chatClient.prompt()
                .user(qus)
                .advisors(ad -> ad.param(ChatMemory.CONVERSATION_ID, sessionId))
                .advisors(ragAdvisor)
                .call()
                .content();
        return ans;
    }

    @GetMapping("/stream")
    public Flux<String> stream(@RequestParam String qus,
                               @RequestParam String sessionId) {
        return chatClient.prompt()
                .user(qus)
                .advisors(ad -> ad.param(ChatMemory.CONVERSATION_ID, sessionId))
                .advisors(ragAdvisor)
                .stream()
                .content();
    }
}
