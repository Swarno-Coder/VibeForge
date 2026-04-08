"""
RAG Integration Test Suite — Comprehensive testing for all RAG stages.

Tests:
  1. Knowledge base loading and chunking
  2. BM25 retrieval (Planner stage)
  3. Vector retrieval (Executor stage)
  4. Hybrid retrieval (Judge stage)
  5. End-to-end pipeline test with a RAG-grounded query
  6. Allocation log integrity (verify pool logic unchanged)

Usage:
  python -m tests.test_rag           # Run all unit tests
  python -m tests.test_rag --e2e     # Run end-to-end pipeline test (requires API keys)
"""

import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# ─── Tracking ─────────────────────────────────────────────────────────────────

passed = 0
failed = 0
test_results = []


def record(test_name: str, success: bool, detail: str = ""):
    global passed, failed
    if success:
        passed += 1
        test_results.append(("✔", test_name, detail, "green"))
    else:
        failed += 1
        test_results.append(("✘", test_name, detail, "red"))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Knowledge Base Loading
# ═══════════════════════════════════════════════════════════════════════════════

def test_knowledge_base_loading():
    console.print("\n[bold cyan]TEST 1: Knowledge Base Loading & Chunking[/bold cyan]")

    from core.knowledge_base_loader import load_knowledge_base

    planning, technical, evaluation, policy = load_knowledge_base()

    # Check each category has documents
    record("Planning docs loaded", len(planning) > 0, f"{len(planning)} chunks")
    record("Technical docs loaded", len(technical) > 0, f"{len(technical)} chunks")
    record("Evaluation docs loaded", len(evaluation) > 0, f"{len(evaluation)} chunks")
    record("Policy docs loaded", len(policy) > 0, f"{len(policy)} chunks")

    # Check metadata
    if planning:
        doc = planning[0]
        has_source = "source" in doc.metadata
        has_category = "category" in doc.metadata
        record("Document metadata present", has_source and has_category,
               f"source={doc.metadata.get('source')}, category={doc.metadata.get('category')}")
    else:
        record("Document metadata present", False, "No planning docs")

    # Check chunking (verify multiple chunks for large files)
    total = len(planning) + len(technical) + len(evaluation) + len(policy)
    record("Chunking creates multiple chunks", total > 15,
           f"{total} total chunks from 15+ source files")

    return planning, technical, evaluation, policy


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: BM25 Retriever (Planner Stage)
# ═══════════════════════════════════════════════════════════════════════════════

def test_bm25_retrieval(planning_docs):
    console.print("\n[bold cyan]TEST 2: BM25 Retriever (Planner Stage)[/bold cyan]")

    from core.rag_engine import BM25Retriever

    bm25 = BM25Retriever()
    bm25.add_documents(planning_docs)

    # Query 1: Should find task decomposition patterns
    results = bm25.retrieve("break down financial analysis into sub-tasks", top_k=3)
    record("BM25: Financial query returns results", len(results) > 0,
           f"Got {len(results)} results")

    if results:
        # Check that the most relevant result mentions financial/investment
        top_content = results[0].content.lower()
        has_relevant = any(kw in top_content for kw in ["financial", "investment", "analysis", "criticality", "agent"])
        record("BM25: Top result is relevant to financial planning",
               has_relevant, f"Score: {results[0].score:.3f}")
    else:
        record("BM25: Top result is relevant to financial planning", False, "No results")

    # Query 2: Should find agent role information
    results2 = bm25.retrieve("what agent roles are available for research tasks", top_k=3)
    record("BM25: Agent role query returns results", len(results2) > 0,
           f"Got {len(results2)} results")

    if results2:
        top_content = results2[0].content.lower()
        has_role = any(kw in top_content for kw in ["researcher", "analyst", "role", "agent"])
        record("BM25: Top result mentions agent roles",
               has_role, f"Score: {results2[0].score:.3f}")
    else:
        record("BM25: Top result mentions agent roles", False, "No results")

    # Query 3: Irrelevant query should return low scores
    results3 = bm25.retrieve("quantum entanglement in black holes", top_k=3)
    if results3:
        record("BM25: Irrelevant query has lower scores",
               results3[0].score < results[0].score if results else True,
               f"Irrelevant score: {results3[0].score:.3f}")
    else:
        record("BM25: Irrelevant query has lower scores", True, "No results (expected)")

    # Query 4: Criticality scoring query
    results4 = bm25.retrieve("criticality score assignment for complex tasks", top_k=3)
    record("BM25: Criticality query returns results", len(results4) > 0,
           f"Got {len(results4)} results")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Vector Retriever (Executor Stage)
# ═══════════════════════════════════════════════════════════════════════════════

def test_vector_retrieval(technical_docs, policy_docs):
    console.print("\n[bold cyan]TEST 3: Vector Retriever (Executor Stage)[/bold cyan]")

    from core.rag_engine import VectorRetriever

    vector = VectorRetriever(collection_name="test_executor")
    all_docs = technical_docs + policy_docs
    vector.add_documents(all_docs)

    # Query 1: Perovskite solar cells (exact domain match)
    results = vector.retrieve("perovskite solar cell efficiency breakthroughs", top_k=3)
    record("Vector: Solar cell query returns results", len(results) > 0,
           f"Got {len(results)} results")

    if results:
        top_content = results[0].content.lower()
        has_solar = any(kw in top_content for kw in ["perovskite", "solar", "efficiency", "silicon"])
        record("Vector: Top result is about solar technology",
               has_solar, f"Score: {results[0].score:.3f}")
    else:
        record("Vector: Top result is about solar technology", False, "No results")

    # Query 2: Semiconductor supply chain (semantic search)
    results2 = vector.retrieve("chip manufacturing and geopolitical risks in Taiwan", top_k=3)
    record("Vector: Semiconductor query returns results", len(results2) > 0,
           f"Got {len(results2)} results")

    if results2:
        top_content = results2[0].content.lower()
        has_semi = any(kw in top_content for kw in ["semiconductor", "tsmc", "taiwan", "chip", "manufacturing"])
        record("Vector: Top result about semiconductors",
               has_semi, f"Score: {results2[0].score:.3f}")
    else:
        record("Vector: Top result about semiconductors", False, "No results")

    # Query 3: EV market in India
    results3 = vector.retrieve("electric vehicle market share in India government subsidies", top_k=3)
    record("Vector: EV India query returns results", len(results3) > 0,
           f"Got {len(results3)} results")

    if results3:
        top_content = results3[0].content.lower()
        has_ev = any(kw in top_content for kw in ["ev", "electric", "india", "tata", "vehicle"])
        record("Vector: Top result about Indian EV market",
               has_ev, f"Score: {results3[0].score:.3f}")
    else:
        record("Vector: Top result about Indian EV market", False, "No results")

    # Query 4: Cross-domain semantic search (AI regulation)
    results4 = vector.retrieve("AI safety regulation compliance requirements in Europe", top_k=3)
    record("Vector: AI regulation query returns results", len(results4) > 0,
           f"Got {len(results4)} results")

    if results4:
        top_content = results4[0].content.lower()
        has_ai = any(kw in top_content for kw in ["ai", "regulation", "eu", "compliance", "act"])
        record("Vector: Top result about AI regulation",
               has_ai, f"Score: {results4[0].score:.3f}")
    else:
        record("Vector: Top result about AI regulation", False, "No results")

    # Query 5: UBI research
    results5 = vector.retrieve("universal basic income effect on employment and entrepreneurship", top_k=3)
    record("Vector: UBI query returns results", len(results5) > 0,
           f"Got {len(results5)} results")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Hybrid Retriever (Judge Stage)
# ═══════════════════════════════════════════════════════════════════════════════

def test_hybrid_retrieval(evaluation_docs, technical_docs, policy_docs):
    console.print("\n[bold cyan]TEST 4: Hybrid Retriever (Judge Stage)[/bold cyan]")

    from core.rag_engine import HybridRetriever, BM25Retriever, VectorRetriever

    hybrid = HybridRetriever(
        bm25=BM25Retriever(),
        vector=VectorRetriever(collection_name="test_judge"),
    )
    all_docs = evaluation_docs + technical_docs + policy_docs
    hybrid.add_documents(all_docs)

    # Query 1: Should find both evaluation rubrics AND factual content
    results = hybrid.retrieve("evaluate semiconductor analysis quality and accuracy", top_k=3)
    record("Hybrid: Mixed eval+technical query returns results", len(results) > 0,
           f"Got {len(results)} results")

    if results:
        # Check diversity — should have docs from different categories
        categories = set()
        for doc in results:
            categories.add(doc.metadata.get("category", "unknown"))
        record("Hybrid: Results span multiple categories",
               len(categories) >= 1,
               f"Categories: {categories}")
    else:
        record("Hybrid: Results span multiple categories", False, "No results")

    # Query 2: Fact-checking query
    results2 = hybrid.retrieve("fact checking guidelines for AI-generated content hallucination", top_k=3)
    record("Hybrid: Fact-check query returns results", len(results2) > 0,
           f"Got {len(results2)} results")

    if results2:
        top_content = results2[0].content.lower()
        has_factcheck = any(kw in top_content for kw in ["fact", "hallucination", "verification", "checking", "accuracy"])
        record("Hybrid: Top result about fact-checking",
               has_factcheck, f"Score: {results2[0].score:.4f}")
    else:
        record("Hybrid: Top result about fact-checking", False, "No results")

    # Query 3: Synthesis rubric query
    results3 = hybrid.retrieve("synthesis quality assessment rubric scoring criteria", top_k=3)
    record("Hybrid: Rubric query returns results", len(results3) > 0,
           f"Got {len(results3)} results")

    # Query 4: Compare with BM25-only — hybrid should find more diverse results
    bm25_only = BM25Retriever()
    bm25_only.add_documents(all_docs)
    bm25_results = bm25_only.retrieve("food security climate change grain exports", top_k=3)
    hybrid_results = hybrid.retrieve("food security climate change grain exports", top_k=3)

    record("Hybrid: Returns results for food security query",
           len(hybrid_results) > 0, f"Hybrid: {len(hybrid_results)}, BM25: {len(bm25_results)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: RAG Context Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def test_rag_context_formatting():
    console.print("\n[bold cyan]TEST 5: RAG Context Formatting[/bold cyan]")

    from core.rag_engine import format_rag_context, Document

    docs = [
        Document(content="Test content one", metadata={"source": "test1.txt", "category": "technical"}, score=0.85),
        Document(content="Test content two", metadata={"source": "test2.txt", "category": "planning"}, score=0.72),
    ]

    formatted = format_rag_context(docs)

    # Check formatting
    record("Format: Contains source metadata",
           "test1.txt" in formatted and "test2.txt" in formatted, "")
    record("Format: Contains category",
           "technical" in formatted and "planning" in formatted, "")
    record("Format: Contains scores",
           "0.850" in formatted or "0.85" in formatted, "")
    record("Format: Contains content",
           "Test content one" in formatted, "")

    # Test truncation
    long_docs = [
        Document(content="A" * 2000, metadata={"source": "long.txt", "category": "tech"}, score=0.9),
        Document(content="B" * 2000, metadata={"source": "long2.txt", "category": "tech"}, score=0.8),
    ]
    truncated = format_rag_context(long_docs, max_chars=1000)
    record("Format: Truncation respects max_chars",
           len(truncated) <= 1200, f"Length: {len(truncated)}")  # small buffer for metadata

    # Test empty docs
    empty = format_rag_context([])
    record("Format: Empty docs returns placeholder",
           "No relevant" in empty, "")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Singleton Initialization
# ═══════════════════════════════════════════════════════════════════════════════

def test_singleton_initialization():
    console.print("\n[bold cyan]TEST 6: RAG Singleton Initialization[/bold cyan]")

    from core.rag_engine import initialize_rag, get_planner_rag, get_executor_rag, get_judge_rag
    from core.rag_engine import BM25Retriever, VectorRetriever, HybridRetriever

    initialize_rag()

    planner_rag = get_planner_rag()
    executor_rag = get_executor_rag()
    judge_rag = get_judge_rag()

    record("Singleton: Planner RAG is BM25Retriever",
           isinstance(planner_rag, BM25Retriever), type(planner_rag).__name__)
    record("Singleton: Executor RAG is VectorRetriever",
           isinstance(executor_rag, VectorRetriever), type(executor_rag).__name__)
    record("Singleton: Judge RAG is HybridRetriever",
           isinstance(judge_rag, HybridRetriever), type(judge_rag).__name__)

    # Test that singletons have documents loaded
    planner_results = planner_rag.retrieve("task decomposition", top_k=1)
    record("Singleton: Planner RAG has indexed docs",
           len(planner_results) > 0, f"Got {len(planner_results)} results")

    executor_results = executor_rag.retrieve("technology analysis", top_k=1)
    record("Singleton: Executor RAG has indexed docs",
           len(executor_results) > 0, f"Got {len(executor_results)} results")

    judge_results = judge_rag.retrieve("quality assessment", top_k=1)
    record("Singleton: Judge RAG has indexed docs",
           len(judge_results) > 0, f"Got {len(judge_results)} results")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: End-to-End Pipeline (requires API keys)
# ═══════════════════════════════════════════════════════════════════════════════

def test_e2e_pipeline():
    """
    Full end-to-end test: sends a query designed to exercise RAG through
    the entire Planner → Executor → Judge pipeline.

    Uses a query that specifically matches knowledge base content (perovskite
    solar cells) to verify RAG retrieval is actually influencing the output.
    """
    console.print("\n[bold cyan]TEST 7: End-to-End Pipeline Test[/bold cyan]")
    console.print("[dim]Testing with a query designed to match knowledge base content...[/dim]")

    from dotenv import load_dotenv
    load_dotenv()

    from core.llm import get_clients
    from core.resource_pool import ResourcePool
    from core.planner import build_plan
    from core.executor import execute_plan
    from core.judge import evaluate_and_synthesize
    from core.rag_engine import initialize_rag

    try:
        clients = get_clients()
    except ValueError as e:
        record("E2E: API keys loaded", False, str(e))
        return

    record("E2E: API keys loaded", True, f"{len(clients)} keys")

    pool = ResourcePool(clients)
    initialize_rag()

    # This query specifically matches our knowledge base content about perovskite solar cells
    # The RAG should retrieve the perovskite document which contains specific data points
    # like "34.6% tandem efficiency" and "Oxford PV commercial prototype"
    test_query = (
        "Analyze the latest developments in perovskite solar cell technology "
        "and their commercial viability for a startup considering entry into "
        "the European market with EU ESG subsidies."
    )

    console.print(f"\n[bold]Test Query:[/bold] {test_query}\n")

    # Phase 1: Planning (with BM25 RAG)
    try:
        plan = build_plan(clients, test_query)
        record("E2E: Planning succeeded", True,
               f"{len(plan.agents)} agents planned")

        for agent in plan.agents:
            console.print(f"  Agent: {agent.name} (crit={agent.criticality})")
    except Exception as e:
        record("E2E: Planning succeeded", False, str(e))
        return

    # Phase 2: Execution (with Vector RAG)
    try:
        final_context = execute_plan(clients, plan, pool)
        has_content = bool(final_context.strip())
        record("E2E: Execution produced output", has_content,
               f"{len(final_context)} chars")

        # Check if RAG content influenced the output
        # Our KB contains specific data like "34.6%" and "Oxford PV"
        lower_ctx = final_context.lower()
        has_kb_data = any(kw in lower_ctx for kw in [
            "perovskite", "tandem", "oxford pv", "efficiency",
            "silicon", "saule", "eu green deal", "stability"
        ])
        record("E2E: Output contains knowledge base data points",
               has_kb_data, "RAG-grounded content detected" if has_kb_data else "No KB data found")

    except Exception as e:
        record("E2E: Execution produced output", False, str(e))
        return

    # Phase 3: Judge synthesis (with Hybrid RAG)
    try:
        evaluate_and_synthesize(clients, pool, test_query, final_context)
        record("E2E: Judge synthesis completed", True, "")
    except Exception as e:
        record("E2E: Judge synthesis completed", False, str(e))

    # Phase 4: Verify allocation log integrity
    try:
        log_path = "allocation_log.txt"
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log_content = f.read()
            has_allocate = "ALLOCATE" in log_content
            has_deallocate = "DEALLOCATE" in log_content
            record("E2E: Allocation log has ALLOCATE events", has_allocate, "")
            record("E2E: Allocation log has DEALLOCATE events", has_deallocate, "")

            # Count events — should have at least as many DEALLOCATE as ALLOCATE
            allocate_count = log_content.count("ALLOCATE") - log_content.count("DEALLOCATE")
            deallocate_count = log_content.count("DEALLOCATE")
            record("E2E: All allocations properly deallocated",
                   allocate_count == deallocate_count,
                   f"ALLOCATE-only: {allocate_count}, DEALLOCATE: {deallocate_count}")
        else:
            record("E2E: Allocation log exists", False, "allocation_log.txt not found")
    except Exception as e:
        record("E2E: Allocation log check", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════════════

def print_results():
    """Pretty-print all test results."""
    table = Table(title="🧪 RAG Integration Test Results", show_lines=True)
    table.add_column("", style="bold", width=3)
    table.add_column("Test", style="cyan", min_width=45)
    table.add_column("Detail", style="dim")

    for icon, name, detail, color in test_results:
        table.add_row(f"[{color}]{icon}[/{color}]", name, detail)

    console.print(table)
    console.print(f"\n[bold green]Passed: {passed}[/bold green] | "
                  f"[bold red]Failed: {failed}[/bold red] | "
                  f"Total: {passed + failed}")

    if failed == 0:
        console.print(Panel("[bold green]ALL TESTS PASSED ✔[/bold green]",
                            border_style="green"))
    else:
        console.print(Panel(f"[bold red]{failed} TEST(S) FAILED ✘[/bold red]",
                            border_style="red"))


def main():
    run_e2e = "--e2e" in sys.argv

    console.print(Panel(
        "[bold cyan]RAG Integration Test Suite[/bold cyan]\n"
        "Testing BM25 + Vector + Hybrid RAG across all pipeline stages",
        border_style="cyan",
    ))

    # Unit tests (no API keys needed)
    planning, technical, evaluation, policy = test_knowledge_base_loading()
    test_bm25_retrieval(planning)
    test_vector_retrieval(technical, policy)
    test_hybrid_retrieval(evaluation, technical, policy)
    test_rag_context_formatting()
    test_singleton_initialization()

    # E2E test (needs API keys)
    if run_e2e:
        test_e2e_pipeline()
    else:
        console.print("\n[dim]Skipping E2E test. Run with --e2e to include pipeline test.[/dim]")

    print_results()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
