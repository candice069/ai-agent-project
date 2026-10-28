from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.core.agent import AIAgent
from app.core.feedback import FeedbackStore
from app.services.llm_service import get_llm_service

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None

class SourceInfo(BaseModel):
    """Source information model"""
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    message_id: str
    response: str
    sources: List[SourceInfo] = Field(default_factory=list)
    session_id: str
    prompt: Dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    """User assessment of a previously generated response."""

    message_id: str
    session_id: str
    rating: int = Field(ge=-1, le=1)
    reason: Optional[str] = Field(default=None, max_length=1000)
    corrected_answer: Optional[str] = Field(default=None, max_length=10000)


class FeedbackResponse(BaseModel):
    message_id: str
    status: str


def get_feedback_store() -> FeedbackStore:
    return FeedbackStore()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_service=Depends(get_llm_service),
    feedback_store: FeedbackStore = Depends(get_feedback_store),
):
    """
    Process a chat message and return a response

    - **message**: The user's message
    - **session_id**: Session identifier for conversation context (optional)
    """
    # Generate a session ID if not provided
    session_id = request.session_id or str(uuid4())

    # Create AI agent
    agent = AIAgent(llm_service)

    try:
        # Process message
        process_args = {"message": request.message, "session_id": session_id}
        if request.metadata_filter is not None:
            process_args["metadata_filter"] = request.metadata_filter
        response, sources = await agent.process_message(**process_args)
        message_id = str(uuid4())
        prompt_metadata = getattr(agent, "last_prompt_metadata", {})
        if not isinstance(prompt_metadata, dict):
            prompt_metadata = {}
        feedback_store.record_response(
            message_id=message_id,
            session_id=session_id,
            question=request.message,
            response=response,
            sources=sources,
            prompt_metadata=prompt_metadata,
        )

        return ChatResponse(
            message_id=message_id,
            response=response,
            sources=sources,
            session_id=session_id,
            prompt=prompt_metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    feedback_store: FeedbackStore = Depends(get_feedback_store),
):
    """Store feedback against a response and its exact prompt version."""
    try:
        feedback = feedback_store.submit_feedback(
            message_id=request.message_id,
            session_id=request.session_id,
            rating=request.rating,
            reason=request.reason,
            corrected_answer=request.corrected_answer,
        )
        if feedback is None:
            raise HTTPException(status_code=404, detail="Response not found")
        return FeedbackResponse(message_id=request.message_id, status="recorded")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error recording feedback: {exc}",
        )


@router.delete("/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a session

    - **session_id**: Session identifier
    """
    try:
        agent = AIAgent(get_llm_service())
        agent.memory_store.clear_history(session_id)
        return {"message": f"Chat history cleared for session {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")
