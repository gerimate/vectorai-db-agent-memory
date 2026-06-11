import logging
import math
import time
import uuid

from actian_vectorai import (
    VectorAIClient,
    VectorParams,
    Distance,
    HnswConfigDiff,
    PointStruct,
    Field,
    FilterBuilder,
)

log = logging.getLogger(__name__)

COLLECTION = "agent_memory"
EMBED_DIM = 384

WEIGHT_SIM = 0.60
WEIGHT_IMPORTANCE = 0.20
WEIGHT_RECENCY = 0.15
WEIGHT_FREQ = 0.05

RECENCY_HALFLIFE_HOURS = 168.0


class MemoryStore:
    def __init__(self, url: str = "localhost:6574"):
        self.client = VectorAIClient(url)
        self.client.connect()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        self.client.collections.get_or_create(
            name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.Cosine),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        )

    @staticmethod
    def _new_id() -> int:
        return uuid.uuid4().int % (2**63 - 1)

    @staticmethod
    def _score(
        sim: float,
        importance: float,
        timestamp: float,
        access_count: int,
    ) -> float:
        now = time.time()
        age_hours = (now - timestamp) / 3600.0
        recency = math.exp(-age_hours / RECENCY_HALFLIFE_HOURS)
        freq = min(access_count / 10.0, 1.0)
        return (
            WEIGHT_SIM * sim
            + WEIGHT_IMPORTANCE * importance
            + WEIGHT_RECENCY * recency
            + WEIGHT_FREQ * freq
        )

    def remember(
        self,
        content: str,
        vector: list[float],
        session_id: str,
        memory_type: str = "episode",
        importance: float = 0.5,
    ) -> str:
        memory_id = self._new_id()
        self.client.points.upsert(
            COLLECTION,
            [
                PointStruct(
                    id=memory_id,
                    vector=vector,
                    payload={
                        "content": content,
                        "session_id": session_id,
                        "memory_type": memory_type,
                        "timestamp": time.time(),
                        "importance": importance,
                        "access_count": 0,
                        "last_accessed": time.time(),
                    },
                )
            ],
        )
        return str(memory_id)

    def recall(
        self,
        query_vector: list[float],
        limit: int = 5,
        session_id: str | None = None,
        memory_type: str | None = None,
        min_importance: float | None = None,
        score_threshold: float = 0.3,
        max_age_days: float | None = None,
    ) -> list[dict]:
        fb = FilterBuilder()
        has_filter = False

        if session_id:
            fb = fb.must(Field("session_id").eq(session_id))
            has_filter = True
        if memory_type:
            fb = fb.must(Field("memory_type").eq(memory_type))
            has_filter = True
        if min_importance is not None:
            fb = fb.must(Field("importance").gte(min_importance))
            has_filter = True
        if max_age_days is not None:
            cutoff = time.time() - 86400 * max_age_days
            fb = fb.must(Field("timestamp").gte(cutoff))
            has_filter = True

        filter_ = fb.build() if has_filter else None

        results = self.client.points.search(
            COLLECTION,
            vector=query_vector,
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold,
            filter=filter_,
        )

        now = time.time()
        scored: list[dict] = []
        for r in results:
            payload = r.payload or {}
            point_id = r.id

            try:
                new_count = payload.get("access_count", 0) + 1
                self.client.points.set_payload(
                    COLLECTION,
                    payload={"access_count": new_count, "last_accessed": now},
                    ids=[point_id],
                )
            except Exception as exc:
                log.warning("Failed to update access metadata for point %s: %s", point_id, exc)
                new_count = payload.get("access_count", 0)

            hybrid = self._score(
                sim=r.score,
                importance=payload.get("importance", 0.5),
                timestamp=payload.get("timestamp", now),
                access_count=new_count,
            )
            scored.append({"id": point_id, "score": hybrid, **payload})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def count(self) -> int:
        return self.client.points.count(COLLECTION)

    def close(self) -> None:
        self.client.close()
