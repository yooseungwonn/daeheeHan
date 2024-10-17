import React, { useState } from "react";

const ChatBot = ({ chatLog, addMessage }) => {
    const [userInput, setUserInput] = useState("");

    const handleFormSubmit = (e) => {
        e.preventDefault();
        if (userInput.trim()) {
            addMessage(userInput); // 사용자 메시지 추가
            setUserInput(""); // 입력 필드 초기화
        }
    };

    return (
        <div id="Chatbot">
            <h1>ChatBot</h1>
            <div id="chat-log">
                {chatLog.map((msg, index) => (
                    <div key={index} className={msg.sender === "사용자" ? "user-message" : "ai-message"}>
                        <strong>{msg.sender}: </strong>
                        {msg.message}
                    </div>
                ))}
            </div>
            <form onSubmit={handleFormSubmit}>
                <input
                    type="text"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    placeholder="메시지를 입력하세요"
                />
                <button type="submit">전송</button>
            </form>
        </div>
    );
};

export default ChatBot;
