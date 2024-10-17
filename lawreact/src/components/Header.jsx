import React, { useEffect, useState } from "react";
import Comant1 from './Comant1';
import Comant2 from './Cal'; 
import Comant3 from './Comant3'; 
import ChatBot from './ChatBot'; 
import axios from "axios"; 
import { v4 as uuid4 } from "uuid";

const headerNav = [
    {
        title: "ChatBot",
        url: "#Comant1",
        toggle: true
    },
    {
        title: "Cal",
        url: "#Cal",
        toggle: true
    },
    {
        title: "기능3",
        url: "#Comant3",
        toggle: true
    }
];

const Header = () => {
    const [activeComponent, setActiveComponent] = useState(null); 
    const [chatbotVisible, setChatbotVisible] = useState(false);
    const [chatbotMessages, setChatbotMessages] = useState([]);

    // 페이지 로드 시 세션 생성
    const [sessionId, setSessionId] = useState("");

    useEffect(() => {setSessionId(uuid4())}, []);

    const handleToggleComponent = (component) => {
        setActiveComponent(prev => (prev === component ? null : component)); 
        if (component === "ChatBot") {
            setChatbotVisible(true);
        } else {
            setChatbotVisible(false);
        }
    };

    const sendMessageToChatbot = async (msg) => {
        setChatbotMessages(prevMessages => [...prevMessages, { sender: "사용자", message: msg }]);

        try {
            console.log("로드중...");
            const response = await axios.post("http://localhost:8000/query", {
                query: msg,
                session_id: sessionId
            });
            console.log("로드완료!");
            setChatbotMessages(prevMessages => [...prevMessages, { sender: "AI", message: response.data.answer }]);
        } catch (error) {
            console.error("Error:", error);
            setChatbotMessages(prevMessages => [...prevMessages, { sender: "AI", message: "서버에 문제가 발생했습니다." }]);
        }
    };

    return (
        <header id="header" role="banner">
            <h1>팀이름</h1>
            <nav>
                <ul className="nav-buttons">
                    {headerNav.map((item, index) => (
                        item.toggle && (
                            <li key={index}>
                                <button onClick={() => handleToggleComponent(item.title)}>
                                    {item.title}
                                </button>
                            </li>
                        )
                    ))}
                </ul>
            </nav>
            {activeComponent === "ChatBot" && <Comant1 sendMessage={sendMessageToChatbot} />}
            {activeComponent === "Cal" && <Comant2 show={activeComponent === "Cal"} />} 
            {activeComponent === "기능3" && <Comant3 />}
            {chatbotVisible && <ChatBot chatLog={chatbotMessages} addMessage={sendMessageToChatbot} />}
        </header>
    );
};

export default Header;
