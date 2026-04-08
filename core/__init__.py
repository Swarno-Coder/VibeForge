"""
Core — Multi-Agent Pipeline Engine.

Modules:
  llm            — API key client factory
  resource_pool  — Semaphore + HashMap resource allocation
  planner        — Task decomposition (structured JSON)
  executor       — Concurrent agent execution
  judge          — Final synthesis & evaluation
  rag_engine     — BM25 / Vector / Hybrid retrieval
  storage        — SQLite + ChromaDB persistence
  cache          — L1 hash + L2 semantic query cache
"""
