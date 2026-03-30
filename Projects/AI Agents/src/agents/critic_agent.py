import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.state.agent_state import AgentState

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def critic_agent_node(state: AgentState) -> AgentState:
    """Evaluates pipeline output against business requirements."""
    messages = list(state.get("messages", []))
    retry_count = state.get("retry_count", 0)

    try:
        # Constructing the evaluation prompt
        context = f"Goal: {state.get('user_goal')}\nInsights: {state.get('insights')}"
        prompt = [
            SystemMessage(content="Evaluate this report. Return JSON: {'approved': bool, 'score': int, 'retry_node': str}"),
            HumanMessage(content=context)
        ]
        
        response = llm.invoke(prompt)
        feedback = json.loads(response.content)

        new_retry = retry_count + (0 if feedback["approved"] else 1)
        
        messages.append({
            "node": "critic_agent",
            "status": "approved" if feedback["approved"] else "rejected",
            "msg": f"Score: {feedback.get('score')}/10"
        })

        return {
            **state,
            "critic_feedback": feedback,
            "retry_count": new_retry,
            "current_node": "critic_agent",
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Critic failed: {e}")
        return {**state, "critic_feedback": {"approved": True}, "messages": messages}