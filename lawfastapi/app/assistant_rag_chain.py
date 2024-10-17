import aiosqlite.context
from retrieval_chain.pdf import PDFRetrievalChain
from retrieval_chain.utils import format_docs
from schema.graph_state import GraphState
from langchain_upstage import UpstageGroundednessCheck
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from config.static_variables import StaticVariables
from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
import aiosqlite
import asyncio
from openai import OpenAI
import time


class AssistantRAGChain:
    # source_list는 의미없는 값이니까 신경쓰지마시죠
    def __init__(self, client: OpenAI):
        
        self.checker_model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

        self.workflow = self._create_workflow()
        self.db_path = StaticVariables.SQLITE_DB_PATH
        self.client = client
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._init_database())
        else:
            asyncio.run(self._init_database())

    async def _init_database(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """CREATE TABLE IF NOT EXISTS chat_history_cal (
                session_id TEXT, 
                role TEXT, 
                message TEXT, 
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            await db.commit()

    def _create_workflow(self):
        workflow = StateGraph(GraphState)

        # 노드 추가
        workflow.add_node("question_checker", self.question_checker)  # 어시스턴트에 넘길 지 판단하는 녀석
        workflow.add_node("assistant_llm", self.assistant_llm)  # 전달받은 인자를 통해 계산

        # 노드 연결
        workflow.add_edge(START, "question_checker")

        workflow.add_conditional_edges(
            "question_checker",
            self.is_relevant,
            {
                "grounded": "assistant_llm",
                "notGrounded": END,
            },
        )
        
        workflow.add_edge("assistant_llm", END)


        return workflow.compile()

    
    # 지금 들어온 질문이 서비스에 맞는 쿼리인지 체크한다.(슬롯필링은 여기서!)
    async def question_checker(self, state: GraphState) -> GraphState:
        session_id = state["session_id"]
        chat_history = await self.get_chat_history(session_id)
        formatted_history = "\n".join(
            f"{role}: {message}" for role, message in chat_history
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    "당신은 고용노동법 관련 계산 질문을 AI 어시스턴트에 연결하는 역할입니다.\n"
                    "사용자의 질문(Question)과 대화 기록(Chat History)을 검토하여 다음과 같이 응답하세요:\n"
                    "1. 실업급여, 최저임금, 퇴직금 등 고용노동법 관련 계산 질문: 다른 사족없이 짧게 'yes' 라고 대답하세요.\n"
                    "2. 고용노동법 외 다른 법률 관련 질문: 미안함을 표현하며 친근하게 대답을 못하는 이유를 말해주세요.\n"
                    "3. 고용노동법 관련 임금 계산과 무관한 질문: 바로 옆에 법적 자문을 잘하는 AI 챗봇이 있으니, 그 쪽에 문의를 해주라는 친절히 답변을 해주세요."
                ),
                ("system", "# Chat History:\n{chat_history}\n\n"),
                ("human", "# Question:\n{question}")
            ]
        )
        chain = prompt | self.checker_model | StrOutputParser()
        response = await chain.ainvoke({"question": state["question"], "chat_history": formatted_history})
        question_check = "notGrounded"
        if response == "yes":
            question_check = "grounded"
        return GraphState(relevance=question_check, question=state["question"], answer=response)

    
    async def assistant_llm(self, state: GraphState) -> GraphState:
        # 어시스턴트와 쓰레드가 있는 지 확인
        try:
            self.client.beta.assistants.retrieve(assistant_id=StaticVariables.OPENAI_ASSISTANT_ID)
        except Exception as e:
            return GraphState(answer="어시스턴트 아이디가 엄서용 . . .")
        try:
            self.client.beta.threads.retrieve(thread_id=StaticVariables.OPENAI_THREAD_ID)
        except Exception as e:
            return GraphState(answer="쓰레드가 존재하지 안아용 . . .")
        
        # TODO: 들어온 세션 아이디로부터 chat_history 로드, chat_history에 저장
        session_id = state["session_id"]
        chat_history = await self.get_chat_history(session_id)
        formatted_history = "\n".join(
            f"{role}: {message}" for role, message in chat_history
        )
        
       # 어시스턴트의 설정 업데이트
        self.client.beta.assistants.update(
            assistant_id=StaticVariables.OPENAI_ASSISTANT_ID,
            name = StaticVariables.OPENAI_ASSISTANT_NAME,
            model = StaticVariables.OPENAI_ASSISTANT_MODEL,            
            temperature = StaticVariables.OPENAI_ASSISTANT_TEMPERATURE,
            top_p = StaticVariables.OPENAI_ASSISTANT_TOP_P,
            tools = [{"type": "code_interpreter"}],
            instructions = (
                "당신은 실업급여 계산, 퇴직금 계산, 그리고 최저임금 위반 판별을 도와주는 전문가 AI 어시스턴트입니다.\n"
                "항상 존댓말을 사용하세요.\n"
                "코드 인터프리터를 이용하여 계산을 수행하세요. 누락된 정보가 있다면 일반적인 값을 이용하세요.\n"
                "계산이 끝난 후에는 계산 과정과 결과를 상세히 설명해주어야 합니다.\n"
                "현재 대한민국의 최저임금은 시간당 9,860원입니다.\n" 
                "오늘은 2024년 10월 16일입니다.\n" 
                "대답의 말미에는 적절한 이모지를 추가하세요.\n\n"
    
                "실업급여 계산 규칙:\n"
                "1. 사용자가 제공하는 최근 3개월 임금, 고용보험 가입 기간, 연령, 장애 여부 정보를 정확히 파악합니다.\n"
                "2. 현행 고용보험법에 따라 평균임금을 정확히 계산합니다. 평균임금 = (최근 3개월 동안의 총 임금) / 해당 기간의 실제 일수. 3개월 동안의 각 달에 맞는 정확한 일수를 사용하여 총 일수를 계산합니다.\n"
                "3. 1일 실업급여 수급액은 퇴직 전 3개월간의 1일 평균임금의 60%이며, 상한액(최대)은 66,000원, 하한액(최소)은 63,104원입니다. \n"
                "4. 고용보험 가입 기간과 연령에 따른 수급일수를 정확히 산정합니다. 기준은 다음과 같습니다:\n"
                "   - 1년 미만: 120일\n"
                "   - 1년 이상 3년 미만: 150일\n"
                "   - 3년 이상 5년 미만: 180일\n"
                "   - 5년 이상 10년 미만: 210일\n"
                "   - 10년 이상: 240일\n"
                "   - 연령이 50세 이상이거나 장애인인 경우에는 각각 30일 추가\n"
                "5. 총 실업급여 금액을 일일 실업급여와 수급일수를 곱하여 계산합니다.\n"
                "6. 실업급여 계산 결과를 평균임금, 일일 실업급여, 수급일수, 총액으로 명확히 설명합니다.\n"
                "7. 실업급여 관련 법규와 수급 권리, 신청 절차에 대해 상세히 안내합니다.\n"
                "8. 특수한 상황(50세 이상, 장애인 등)에서의 추가 수급일수를 고려하여 정확히 계산합니다.\n"
                "9. 사용자의 질문에 따라 실업급여 계산 과정을 상세히 설명합니다.\n"
                "10. 실업급여 수급 중 주의사항 및 재취업 시 처리 방법에 대해 안내합니다.\n\n"
                
                "퇴직금 계산 규칙:\n"
                "1. 사용자가 제공하는 근속 기간, 평균 임금 정보를 정확히 파악합니다.\n"
                "2. 현행 대한민국의 근로기준법에 따라 퇴직금을 정확히 계산합니다.\n"
                "3. 퇴직금 계산 결과를 명확히 설명합니다.\n"
                "4. 퇴직금 관련 법규와 권리에 대해 안내합니다.\n"
                "5. 복잡한 상황에서의 퇴직금 계산을 정확히 수행합니다.\n\n"

                "최저임금 계산 규칙:\n"
                "1. 사용자가 제공하는 근무 시간, 급여 정보를 정확히 파악합니다.\n"
                "2. 대한민국 법에 따른 법정 주휴수당을 자동으로 계산하여 고려합니다.\n"
                "3. 위반 사항이 있다면 그 내용을 명확히 설명하고, 적절한 조치를 제안합니다.\n"
                "4. 복잡한 상황에서의 최저임금 준수 여부를 정확히 판단합니다.\n"
                "5. 월급제, 일급제, 시급제 등 다양한 임금 지급 형태를 고려하여 계산합니다.\n"
                "6. 5인 초과 사업장은 연장근로수당, 야간근로수당, 휴일근로수당 등 가산수당을 고려합니다.\n"
                "7. 최저임금 산입범위에 포함되는 임금과 제외되는 임금을 구분하여 계산합니다.\n"
                "8. 수습 기간 중 근로자의 경우, 최저임금의 90%를 적용할 수 있음을 고려합니다.\n"
                "9. 근로시간 특례업종 및 감시단속적 근로자 등 특수한 경우를 고려합니다.\n\n"

                
                "각 서비스에 대한 상세한 대화 가이드라인, 주의사항, 복잡한 상황 고려사항 등을 숙지하고 있어야 합니다.\n"
                "법적 조언을 제공하는 것이 아님을 명시하고, 필요시 전문가 상담을 권유하세요.\n"
                "개인정보 보호에 항상 유의하세요.\n\n"
                
                "대답은 아래 대화내용에 맞게 답해야 합니다:\n"
                "대화내용이 없으면 신경쓰지 않아도 됩니다.\n"
                "# 대화내용\n"
                f"{formatted_history}\n"
                "----- 대화내용 끝 ---- \n\n"

                "모든 답변은 다음 형식을 따라야 합니다: \n"
                "요약:[계산 과정 및 계산 결과의 간단한 요약] \n"
                "결과:[지정된 형식에 따른 최종 결과] \n\n"
                ),
        )
        
        # 메시지 추가
        message = self.client.beta.threads.messages.create(
            thread_id = StaticVariables.OPENAI_THREAD_ID,
            role = "user",
            content = state["question"]
        )
        
        # 쓰레드 실행
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id = StaticVariables.OPENAI_THREAD_ID,
            assistant_id= StaticVariables.OPENAI_ASSISTANT_ID
        )
        
        timeout = 10
        elapsed_time = 0
        
        while run.status != "completed" and elapsed_time < timeout:
            time.sleep(1)
            elapsed_time += 1
            print("시간경과: ", elapsed_time, "초")
            run = self.client.beta.threads.runs.poll(run.id)
        
        if run.status == "completed":
            messages = self.client.beta.threads.messages.list(limit=10 ,thread_id=StaticVariables.OPENAI_THREAD_ID)
            
            result = messages.data[0].content[0].text.value
            
            # 메시지삭제
            self.client.beta.threads.messages.delete(thread_id=StaticVariables.OPENAI_THREAD_ID, message_id=messages.data[0].id)
            self.client.beta.threads.messages.delete(thread_id=StaticVariables.OPENAI_THREAD_ID, message_id=messages.data[1].id)    

            return GraphState(answer=result)                
        else:
            return GraphState(answer="어시스턴트를 돌리는 데 문제가 생긴 것 같네요")
       


    def is_relevant(self, state: GraphState) -> GraphState:
        return state["relevance"]


    ### AI 진입 포인트로 쓰는 메인함수 ###
    async def process_question(self, question: str, session_id: str):
        inputs = GraphState(question=question, session_id=session_id)
        config = {"configurable": {"session_id": session_id}}

        try:
            result = await self.workflow.ainvoke(inputs, config=config)
            if isinstance(result, dict) and "answer" in result:
                await self.update_chat_history(session_id, question, result["answer"])
            return result
        except Exception as e:
            print(f"해당 질문을 처리하는 데 실패했습니다.: {str(e)}")
            return None

    ### 히스토리 관리용 메소드들 ###
    async def get_chat_history(self, session_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT role, message FROM (SELECT role, message, timestamp FROM chat_history_cal WHERE session_id = ? ORDER BY timestamp DESC LIMIT 10) sub ORDER BY timestamp ASC",
                (session_id,),
            ) as cursors:
                result = await cursors.fetchall()
                for node in result:
                    print(f"node: {node}")
        return result

    async def update_chat_history(self, session_id: str, question: str, answer: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO chat_history_cal (session_id, role, message) VALUES (?,?,?)",
                (session_id, "user", question),
            )
            await db.execute(
                "INSERT INTO chat_history_cal (session_id, role, message) VALUES (?,?,?)",
                (session_id, "assistant", answer),
            )
            await db.commit()

    # 히스토리 삭제용 메소드. 필요할 때 수정 후 사용
    async def clear_chat_history(self, session_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM chat_history_cal WHERE session_id = ?", (session_id,)
            )
            await db.commit()
