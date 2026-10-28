from typing import List, Dict, Tuple, Any, Optional
import logging

from app.core.context_builder import ContextBuilder
from app.core.memory import RedisMemoryStore
from app.core.prompts import get_prompt
from app.core.query_router import QueryRouter
from app.services.serp_service import SerpService
from app.services.qdrant_service import QdrantService
from app.utils.embeddings import get_embedding, batch_get_embeddings
from app.utils.ranking import (
    rank_results,
    filter_results,
    deduplicate_results,
    diversify_results,
    rank_sources,
)
from app.config import (
    MAX_CONTEXT_LENGTH,
    RAG_CANDIDATE_LIMIT,
    RAG_MIN_SCORE,
    RAG_DENSE_WEIGHT,
)

logger = logging.getLogger(__name__)

class AIAgent:
    """Main AI agent that orchestrates the conversation flow and information retrieval"""

    def __init__(
        self,
        llm_service,
        memory_store: Optional[RedisMemoryStore] = None,
        serp_service: Optional[SerpService] = None,
        qdrant_service: Optional[QdrantService] = None,
        query_router: Optional[QueryRouter] = None,
    ):
        """Initialize the AI agent with necessary services

        Args:
            llm_service: Service providing access to the language model
        """
        self.llm = llm_service.get_llm()
        self.memory_store = memory_store or RedisMemoryStore()
        self.serp_service = serp_service or SerpService()
        self.qdrant_service = qdrant_service or QdrantService()
        self.query_router = query_router or QueryRouter()
        self.context_builder = ContextBuilder()
        self.last_prompt_metadata: Dict[str, Any] = {}

    async def process_message(
        self,
        message: str,
        session_id: str,
        metadata_filter: Optional[Any] = None,
        interaction_type: str = "text",
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Process a user message and generate a response

        Args:
            message: User's message
            session_id: Session ID for tracking conversation history

        Returns:
            Tuple containing the response text and list of sources
        """
        try:
            # Retrieve conversation history
            history = self.memory_store.get_history(session_id)
            summary = self.memory_store.get_summary(session_id)

            # Generate embedding for the message
            embedding = get_embedding(message)
            route = self.query_router.route(message)

            # Retrieve relevant context from vector store (RAG)
            rag_results = []
            if route.use_knowledge_base:
                rag_results = self.qdrant_service.hybrid_search(
                    query=message,
                    embedding=embedding,
                    limit=MAX_CONTEXT_LENGTH * 2,
                    candidate_limit=RAG_CANDIDATE_LIMIT,
                    filter_condition=metadata_filter,
                    dense_weight=RAG_DENSE_WEIGHT,
                )
            # Compatibility fallback for adapters that only implement dense
            # search; also keeps the service easy to mock in unit tests.
            if route.use_knowledge_base and not isinstance(rag_results, list):
                rag_results = self.qdrant_service.search(
                    embedding,
                    limit=MAX_CONTEXT_LENGTH * 2,
                    filter_condition=metadata_filter,
                )

            # Search the web if needed
            web_results = []
            if self._needs_web_search(message):
                logger.info(f"Performing web search for query: {message}")
                web_results = self.serp_service.search(message)
                web_results = rank_sources(web_results)
                web_embeddings = batch_get_embeddings([
                    f"{item.get('title', '')} {item.get('content', '')}"
                    for item in web_results
                ])
                for item, item_embedding in zip(web_results, web_embeddings):
                    item["embedding"] = item_embedding

            # Combine and rank all results
            all_results = deduplicate_results(rag_results + web_results)
            ranked_results = rank_results(all_results, embedding)
            filtered_results = diversify_results(
                filter_results(ranked_results, min_score=RAG_MIN_SCORE)
            )

            history_text = history.load_memory_variables({}).get("history", "")
            prompt_context = self.context_builder.build(
                user_input=message,
                history=history_text,
                summary=summary,
                results=filtered_results[:MAX_CONTEXT_LENGTH],
            )
            prompt_name = (
                "voice" if interaction_type == "voice" else route.prompt_type
            )
            # Run the model with an explicitly rendered prompt. This keeps
            # memory persistence under our control and works across LangChain
            # versions without ConversationChain's strict input validation.
            response = await self._generate_response(
                prompt_name=prompt_name,
                input=message,
                context=prompt_context.context,
                history=prompt_context.history,
                summary=prompt_context.summary,
                metadata={
                    "route_reason": route.reason,
                    "estimated_input_tokens": (
                        prompt_context.estimated_input_tokens
                    ),
                },
            )
            history.save_context({"input": message}, {"output": response})

            # Extract source information
            sources = [
                {
                    "title": r.get("title", "Unknown"),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:150] + "..." if len(r.get("content", "")) > 150 else r.get("content", "")
                }
                for r in filtered_results[:MAX_CONTEXT_LENGTH]
            ]

            # Store the interaction in memory
            self.memory_store.add_interaction(session_id, message, response)
            await self._maybe_update_summary(session_id, message, response)

            return response, sources

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise

    async def _generate_response(
        self,
        *,
        prompt_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        **prompt_values: str,
    ) -> str:
        prompt = get_prompt(prompt_name)
        rendered = prompt.template.format(**prompt_values)
        self.last_prompt_metadata = {
            "prompt_name": prompt.name,
            "prompt_version": prompt.version,
            **(metadata or {}),
        }
        return await self._invoke_rendered(rendered)

    async def _invoke_rendered(self, rendered: str) -> str:
        if hasattr(self.llm, "arun"):
            result = await self.llm.arun(rendered)
        elif hasattr(self.llm, "ainvoke"):
            result = await self.llm.ainvoke(rendered)
        else:
            raise TypeError("Configured LLM does not support async invocation")
        return str(getattr(result, "content", result))

    def _needs_web_search(self, message: str) -> bool:
        """Determine if web search is needed for this message

        Args:
            message: User's message

        Returns:
            Boolean indicating if web search should be performed
        """
        return self.query_router.route(message).use_web

    async def _maybe_update_summary(
        self,
        session_id: str,
        user_message: str,
        response: str,
    ) -> None:
        """Update rolling long-term memory after the configured threshold."""
        if self.memory_store.should_summarize(session_id) is not True:
            return
        previous = self.memory_store.get_summary(session_id)
        prompt = get_prompt("summary").template.format(
            previous=previous,
            user_message=user_message,
            response=response,
        )
        try:
            if hasattr(self.llm, "apredict"):
                summary = await self.llm.apredict(text=prompt)
            else:
                summary = f"{previous}\nUser: {user_message}\nAssistant: {response}"
            self.memory_store.set_summary(session_id, str(summary))
        except Exception as exc:
            logger.warning(f"Unable to update conversation summary: {exc}")

    def reset(self, session_id: str) -> None:
        """Reset the agent's state for a session

        Args:
            session_id: Session ID to reset
        """
        self.memory_store.clear_history(session_id)
