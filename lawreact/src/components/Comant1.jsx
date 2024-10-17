import React from "react";

const Comant1 = ({ sendMessage }) => {
    const handleSendMessage = () => {
        if (sendMessage) {
            sendMessage("노동법에 대해 알려줘"); 
        }
    };

    const handleSendMessage2= () =>{
        if (sendMessage){
            sendMessage("임금 계산법에 대해 알려줘")
        }
    }

    return (
        <div className="button-container">
            <button onClick={handleSendMessage}>노동법</button>
            <button onClick={handleSendMessage2}>계산</button>
        </div>
    );
};

export default Comant1;
