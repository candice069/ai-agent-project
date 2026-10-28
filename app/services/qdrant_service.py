import qdrant_client
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from typing import Dict, List, Any, Optional, Union
import uuid
import logging
import re

from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

class QdrantService:
    """Service for Qdrant vector store operations"""

    def __init__(self,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 collection_name: Optional[str] = None):
        """Initialize the Qdrant service

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Vector collection name
        """
        self.host = host or QDRANT_HOST
        self.port = port or QDRANT_PORT
        self.collection_name = collection_name or QDRANT_COLLECTION

        # Initialize the Qdrant client
        self._init_client()

        # Initialize collection if it doesn't exist
        self._init_collection()

    def _init_client(self) -> None:
        """Initialize the Qdrant client"""
        try:
            self.client = qdrant_client.QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise

    def _init_collection(self) -> None:
        """Initialize the vector collection if it doesn't exist"""
        try:
            # Get existing collections
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            # Create the collection if it doesn't exist
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )

                logger.info(f"Collection {self.collection_name} created")
            else:
                logger.info(f"Collection {self.collection_name} exists")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {str(e)}")
            raise

    def add_document(self,
                     document_id: Optional[str] = None,
                     embedding: List[float] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Add a document to the vector store

        Args:
            document_id: Document ID (generated if not provided)
            embedding: Vector embedding
            metadata: Document metadata

        Returns:
            Document ID
        """
        if not embedding:
            raise ValueError("Embedding vector is required")

        if not metadata:
            metadata = {}

        # Generate a document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())

        try:
            # Create point for Qdrant
            point = PointStruct(
                id=document_id,
                vector=embedding,
                payload=metadata
            )

            # Upsert the point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.debug(f"Document {document_id} added to Qdrant")

            return document_id
        except Exception as e:
            logger.error(f"Failed to add document to Qdrant: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents to the vector store

        Args:
            documents: List of document dictionaries with 'id', 'embedding', and 'metadata'

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        try:
            # Prepare points for Qdrant
            points = []
            document_ids = []

            for doc in documents:
                # Check for required fields
                if 'embedding' not in doc:
                    raise ValueError(f"Document missing 'embedding' field: {doc}")

                # Generate an ID if not provided
                doc_id = doc.get('id', str(uuid.uuid4()))
                document_ids.append(doc_id)

                # Create point
                payload = dict(doc.get('metadata', {}))
                payload.update({
                    key: doc[key]
                    for key in ("content", "title", "url")
                    if doc.get(key) is not None
                })
                points.append(PointStruct(
                    id=doc_id,
                    vector=doc['embedding'],
                    payload=payload
                ))

            # Upload in batches (Qdrant recommends max 100 points per batch)
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

            logger.info(f"Added {len(documents)} documents to Qdrant")

            return document_ids
        except Exception as e:
            logger.error(f"Failed to add documents to Qdrant: {str(e)}")
            raise

    def search(self,
               embedding: List[float],
               limit: int = 5,
               filter_condition: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents

        Args:
            embedding: Query vector embedding
            limit: Maximum number of results to return
            filter_condition: Optional filter for the search

        Returns:
            List of dictionaries with content and metadata
        """
        if not embedding:
            raise ValueError("Embedding vector is required")
        filter_condition = self._coerce_filter(filter_condition)

        try:
            # Perform search
            if hasattr(type(self.client), "query_points"):
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=embedding,
                    limit=limit,
                    query_filter=filter_condition,
                ).points
            else:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding,
                    limit=limit,
                    query_filter=filter_condition
                )

            # Format results
            formatted_results = []

            for result in search_result:
                # Extract payload fields
                payload = result.payload
                formatted_result = {
                    "content": payload.get("content", ""),
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                    "source": payload.get("source", "vector_store"),
                    "score": result.score
                }

                # Include all other payload fields
                for key, value in payload.items():
                    if key not in formatted_result:
                        formatted_result[key] = value

                formatted_results.append(formatted_result)

            logger.debug(f"Found {len(formatted_results)} similar documents")

            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {str(e)}")
            return []

    def hybrid_search(
        self,
        query: str,
        embedding: List[float],
        limit: int = 5,
        candidate_limit: int = 20,
        filter_condition: Optional[Any] = None,
        dense_weight: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """Fuse dense-vector and lexical payload rankings with weighted RRF.

        Lexical candidates are read from Qdrant payloads. This keeps one source
        of truth and provides a deterministic sparse-style baseline without
        requiring a second search service.
        """
        dense = self.search(
            embedding,
            limit=candidate_limit,
            filter_condition=filter_condition,
        )
        lexical = self._lexical_search(
            query,
            limit=candidate_limit,
            filter_condition=filter_condition,
        )

        fused: Dict[str, Dict[str, Any]] = {}
        rrf_k = 60
        for weight, results in (
            (dense_weight, dense),
            (1.0 - dense_weight, lexical),
        ):
            for rank, result in enumerate(results, start=1):
                key = result.get("url") or (
                    f"{result.get('source', '')}:{result.get('title', '')}:"
                    f"{result.get('content', '')[:120]}"
                )
                if key not in fused:
                    fused[key] = dict(result)
                    fused[key]["score"] = 0.0
                fused[key]["score"] += weight / (rrf_k + rank)
                fused[key]["retrieval"] = "hybrid"

        ranked = sorted(
            fused.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )[:limit]
        max_score = max((item["score"] for item in ranked), default=1.0)
        for item in ranked:
            item["score"] = item["score"] / max_score if max_score else 0.0
        return ranked

    def _lexical_search(
        self,
        query: str,
        limit: int,
        filter_condition: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Rank Qdrant payloads by normalized token overlap."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return []
        filter_condition = self._coerce_filter(filter_condition)
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=max(limit * 5, 50),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning(f"Lexical retrieval unavailable: {exc}")
            return []

        results = []
        for point in points:
            payload = point.payload or {}
            content = f"{payload.get('title', '')} {payload.get('content', '')}"
            tokens = set(re.findall(r"\w+", content.lower()))
            if not tokens:
                continue
            overlap = len(query_tokens & tokens) / len(query_tokens)
            if overlap <= 0:
                continue
            item = dict(payload)
            item.setdefault("source", "vector_store")
            item["score"] = overlap
            item["retrieval"] = "lexical"
            results.append(item)
        return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]

    @staticmethod
    def _coerce_filter(filter_condition: Optional[Any]) -> Optional[Any]:
        """Convert a simple metadata mapping into a Qdrant filter."""
        if not filter_condition or not isinstance(filter_condition, dict):
            return filter_condition
        return Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_condition.items()
            ]
        )

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store

        Args:
            document_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[document_id]
            )

            logger.debug(f"Document {document_id} deleted from Qdrant")

            return True
        except Exception as e:
            logger.error(f"Failed to delete document from Qdrant: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection

        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )

            # Format the information
            info = {
                "name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count
            }

            return info
        except Exception as e:
            logger.error(f"Failed to get Qdrant collection info: {str(e)}")
            return {}

# Dependency for FastAPI
def get_qdrant_service():
    """Get a Qdrant service instance for dependency injection"""
    return QdrantService()
