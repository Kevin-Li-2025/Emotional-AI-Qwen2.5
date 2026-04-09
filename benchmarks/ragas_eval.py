"""
RAGAS Evaluation Pipeline — Lin Xia Memory Quality Assessment

Runs a full RAGAS evaluation against Lin Xia's RAG memory pipeline.
Measures:
  - Faithfulness:      Does Lin Xia's response stay true to retrieved memories?
  - Answer Relevancy:  Is the response relevant to the user's question?
  - Context Precision:  Are higher-ranked memories more useful?
  - Context Recall:     Did retrieval miss any critical memories?

Usage:
  python benchmarks/ragas_eval.py                    # Run full evaluation
  python benchmarks/ragas_eval.py --category fact_recall  # Single category
  python benchmarks/ragas_eval.py --no-llm           # Skip LLM judge, test infra only

Output:
  benchmarks/ragas_report.md    — Evaluation report with scores
  benchmarks/ragas_raw.json     — Raw results for further analysis
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.memory_store import MemoryStore, Memory, MemoryType
from memory.memory_retriever import MemoryRetriever
from benchmarks.ragas_testset import EVAL_CASES, EvalCase, get_all_categories, summary


# ---------------------------------------------------------------------------
# Evaluation Engine (RAGAS-compatible, self-contained)
# ---------------------------------------------------------------------------

class EmotionalRAGEvaluator:
    """
    Evaluates Lin Xia's RAG pipeline using RAGAS-style metrics.
    
    Supports two modes:
    1. Full mode:   Uses RAGAS library + LLM-as-a-judge (DeepSeek/OpenAI)
    2. Infra mode:  Tests retrieval pipeline only (no external LLM needed)
    """

    def __init__(self, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge
        self.results: list[dict] = []

        # Set up a fresh memory store for evaluation (isolated)
        self.eval_db_path = "./memory_db/eval_temp"
        self.store = MemoryStore(
            db_path=self.eval_db_path,
            collection_name="ragas_eval"
        )
        self.retriever = MemoryRetriever(self.store, recency_weight=0.2)

        # LLM for response generation
        self.llm = None

    def _load_llm(self):
        """Load the local LLM for response generation."""
        if self.llm is not None:
            return

        try:
            from llama_cpp import Llama
            model_candidates = [
                "emotional-model-output/linxia-dpo-q8_0.gguf",
                "emotional-model-output/linxia-emotional-v2-q8_0.gguf",
                "emotional-model-output/linxia-q4_k_m.gguf",
            ]
            model_path = next((p for p in model_candidates if os.path.exists(p)), None)
            if model_path:
                print(f"  Loading LLM: {os.path.basename(model_path)}")
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048, n_gpu_layers=-1, verbose=False,
                )
            else:
                print("  [WARNING] No GGUF model found. Using template responses.")
        except ImportError:
            print("  [WARNING] llama_cpp not available. Using template responses.")

    def _seed_memories(self, case: EvalCase):
        """Inject seed memories for a test case."""
        # Clear previous memories
        self.store.clear()

        for i, mem_text in enumerate(case.seed_memories):
            memory = Memory(
                content=mem_text,
                memory_type=MemoryType.FACT,
                importance=7.0,
                emotional_valence=0.0,
                timestamp=time.time() - (len(case.seed_memories) - i) * 3600,
            )
            self.store.add_memory(memory)

    def _retrieve(self, question: str, n_results: int = 3) -> list[str]:
        """Retrieve memories for a question."""
        return self.retriever.retrieve(question, n_results=n_results)

    def _generate_response(self, question: str, contexts: list[str]) -> str:
        """Generate Lin Xia's response using the LLM with retrieved context."""
        self._load_llm()

        if self.llm is None:
            # Template fallback for infra-only testing
            if contexts:
                return f"根据我的记忆：{contexts[0]}"
            return "我不太记得了..."

        context_str = "\n".join(f"- {c}" for c in contexts)
        system = (
            "你是林夏，一个真实的、有情绪的女生。"
            "根据你记忆中的以下信息来回答用户的问题。"
            "如果记忆中没有相关信息，诚实地说你不太记得了。\n\n"
            f"[你的记忆]\n{context_str}"
        )

        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        output = self.llm(
            prompt, max_tokens=200, stop=["<|im_end|>"],
            temperature=0.5, repeat_penalty=1.1,
        )
        response = output["choices"][0]["text"].strip()

        # Strip emotion tags
        import re
        response = re.sub(r'<emotion[^>]*/>', '', response).strip()
        return response

    def evaluate_case(self, case: EvalCase) -> dict:
        """Run a single evaluation case through the pipeline."""
        # 1. Seed memories
        self._seed_memories(case)

        # 2. Retrieve
        contexts = self._retrieve(case.question)

        # 3. Generate
        answer = self._generate_response(case.question, contexts)

        # 4. Compute local metrics (no external LLM needed)
        retrieval_score = self._score_retrieval(
            contexts, case.seed_memories, case.ground_truth
        )
        faithfulness_score = self._score_faithfulness(answer, contexts)
        relevancy_score = self._score_relevancy(answer, case.question)

        result = {
            "question": case.question,
            "ground_truth": case.ground_truth,
            "answer": answer,
            "contexts": contexts,
            "category": case.category,
            "difficulty": case.difficulty,
            "metrics": {
                "context_recall": retrieval_score["recall"],
                "context_precision": retrieval_score["precision"],
                "faithfulness": faithfulness_score,
                "answer_relevancy": relevancy_score,
            },
        }
        self.results.append(result)
        return result

    def _score_retrieval(self, retrieved: list[str], seed: list[str],
                         ground_truth: str) -> dict:
        """
        Score retrieval quality using keyword overlap heuristic.
        (In full RAGAS mode, this is replaced by LLM-as-a-judge.)
        """
        if not retrieved:
            return {"recall": 0.0, "precision": 0.0}

        # Extract keywords from ground truth
        gt_keywords = set()
        for word in ground_truth:
            if '\u4e00' <= word <= '\u9fff':  # Chinese char
                gt_keywords.add(word)
        # Add multi-char keywords
        for seed_mem in seed:
            for word in seed_mem.split():
                if len(word) >= 2:
                    gt_keywords.add(word)

        # Check overlap
        retrieved_text = " ".join(retrieved)
        hits = sum(1 for kw in gt_keywords if kw in retrieved_text)
        recall = hits / max(len(gt_keywords), 1)

        # Precision: how many retrieved docs contain relevant info
        relevant_count = 0
        for ctx in retrieved:
            if any(kw in ctx for kw in gt_keywords if len(kw) >= 2):
                relevant_count += 1
        precision = relevant_count / max(len(retrieved), 1)

        return {
            "recall": round(min(recall, 1.0), 3),
            "precision": round(precision, 3),
        }

    def _score_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        Score whether the answer is faithful to the retrieved contexts.
        Heuristic: keyword overlap between answer and contexts.
        """
        if not contexts or not answer:
            return 0.0

        context_text = " ".join(contexts)

        # Extract answer claims (Chinese characters and meaningful words)
        answer_chars = set()
        for i in range(len(answer) - 1):
            bigram = answer[i:i+2]
            if all('\u4e00' <= c <= '\u9fff' for c in bigram):
                answer_chars.add(bigram)

        if not answer_chars:
            return 0.5  # Can't evaluate

        supported = sum(1 for bg in answer_chars if bg in context_text)
        return round(supported / len(answer_chars), 3)

    def _score_relevancy(self, answer: str, question: str) -> float:
        """
        Score answer relevancy to the question.
        Heuristic: shared meaningful content between Q and A.
        """
        if not answer or not question:
            return 0.0

        # Check if the answer addresses the question topic
        q_chars = set()
        for i in range(len(question) - 1):
            bigram = question[i:i+2]
            if all('\u4e00' <= c <= '\u9fff' for c in bigram):
                q_chars.add(bigram)

        if not q_chars:
            return 0.5

        # A relevant answer should reference the question's topic
        addressed = sum(1 for bg in q_chars if bg in answer)
        base_score = min(addressed / max(len(q_chars), 1), 1.0)

        # Bonus: answer is not too short and not a cop-out
        if len(answer) < 5:
            base_score *= 0.5
        if "不记得" in answer or "不知道" in answer:
            base_score *= 0.3

        return round(base_score, 3)

    def run_full_evaluation(self, cases: list[EvalCase] = None,
                            category: str = None) -> dict:
        """Run evaluation on all or filtered test cases."""
        if cases is None:
            cases = EVAL_CASES
        if category:
            cases = [c for c in cases if c.category == category]

        print(f"\n{'='*60}")
        print(f"RAGAS Evaluation — Lin Xia Memory Quality")
        print(f"{'='*60}")
        print(f"  Test cases: {len(cases)}")
        print(f"  Categories: {len(set(c.category for c in cases))}")
        print(f"  LLM Judge:  {'Yes' if self.use_llm_judge else 'No (heuristic)'}")

        self.results = []
        for i, case in enumerate(cases):
            print(f"\n  [{i+1}/{len(cases)}] [{case.category}] {case.question[:40]}...")
            result = self.evaluate_case(case)
            m = result["metrics"]
            print(f"    Answer:  {result['answer'][:60]}...")
            print(f"    Scores:  F={m['faithfulness']:.2f}  R={m['answer_relevancy']:.2f}  "
                  f"CP={m['context_precision']:.2f}  CR={m['context_recall']:.2f}")

        # Aggregate results
        return self._aggregate()

    def _aggregate(self) -> dict:
        """Compute aggregate metrics across all results."""
        if not self.results:
            return {}

        metrics = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
        }
        by_category = {}
        by_difficulty = {}

        for r in self.results:
            for metric, value in r["metrics"].items():
                metrics[metric].append(value)

            cat = r["category"]
            diff = r["difficulty"]

            if cat not in by_category:
                by_category[cat] = {k: [] for k in metrics}
            for k, v in r["metrics"].items():
                by_category[cat][k].append(v)

            if diff not in by_difficulty:
                by_difficulty[diff] = {k: [] for k in metrics}
            for k, v in r["metrics"].items():
                by_difficulty[diff][k].append(v)

        def avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        aggregate = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(self.results),
            "overall": {k: avg(v) for k, v in metrics.items()},
            "by_category": {
                cat: {k: avg(v) for k, v in cat_metrics.items()}
                for cat, cat_metrics in by_category.items()
            },
            "by_difficulty": {
                diff: {k: avg(v) for k, v in diff_metrics.items()}
                for diff, diff_metrics in by_difficulty.items()
            },
        }

        return aggregate

    def generate_report(self, aggregate: dict, output_path: str = "benchmarks/ragas_report.md"):
        """Generate a markdown evaluation report."""
        lines = []
        lines.append("# RAGAS Evaluation Report — Lin Xia Memory Quality\n")
        lines.append(f"**Generated:** {aggregate.get('timestamp', 'N/A')}  ")
        lines.append(f"**Test Cases:** {aggregate.get('total_cases', 0)}  ")
        lines.append(f"**Evaluation Mode:** {'LLM-as-a-Judge' if self.use_llm_judge else 'Heuristic'}\n")

        # Overall scores
        overall = aggregate.get("overall", {})
        lines.append("## Overall Scores\n")
        lines.append("| Metric | Score | Description |")
        lines.append("|--------|-------|-------------|")
        lines.append(f"| **Faithfulness** | **{overall.get('faithfulness', 0):.1%}** | 回答是否忠于检索到的记忆 |")
        lines.append(f"| **Answer Relevancy** | **{overall.get('answer_relevancy', 0):.1%}** | 回答与问题的相关性 |")
        lines.append(f"| **Context Precision** | **{overall.get('context_precision', 0):.1%}** | 检索排序质量 |")
        lines.append(f"| **Context Recall** | **{overall.get('context_recall', 0):.1%}** | 关键记忆召回率 |")

        # Composite score
        scores = [v for v in overall.values() if isinstance(v, (int, float))]
        composite = sum(scores) / len(scores) if scores else 0
        lines.append(f"\n**Composite Score: {composite:.1%}**\n")

        if composite >= 0.8:
            lines.append("> [!TIP]\n> 🎉 Excellent! Memory pipeline is performing well.\n")
        elif composite >= 0.6:
            lines.append("> [!NOTE]\n> ⚡ Good baseline. Room for improvement in retrieval.\n")
        else:
            lines.append("> [!WARNING]\n> ⚠️ Memory pipeline needs significant improvement.\n")

        # By category
        by_cat = aggregate.get("by_category", {})
        if by_cat:
            lines.append("## Scores by Category\n")
            lines.append("| Category | Faithfulness | Relevancy | Precision | Recall |")
            lines.append("|----------|-------------|-----------|-----------|--------|")
            for cat, scores in sorted(by_cat.items()):
                lines.append(
                    f"| {cat} "
                    f"| {scores.get('faithfulness', 0):.1%} "
                    f"| {scores.get('answer_relevancy', 0):.1%} "
                    f"| {scores.get('context_precision', 0):.1%} "
                    f"| {scores.get('context_recall', 0):.1%} |"
                )

        # By difficulty
        by_diff = aggregate.get("by_difficulty", {})
        if by_diff:
            lines.append("\n## Scores by Difficulty\n")
            lines.append("| Difficulty | Faithfulness | Relevancy | Precision | Recall |")
            lines.append("|------------|-------------|-----------|-----------|--------|")
            for diff in ["easy", "normal", "hard"]:
                if diff in by_diff:
                    scores = by_diff[diff]
                    lines.append(
                        f"| {diff} "
                        f"| {scores.get('faithfulness', 0):.1%} "
                        f"| {scores.get('answer_relevancy', 0):.1%} "
                        f"| {scores.get('context_precision', 0):.1%} "
                        f"| {scores.get('context_recall', 0):.1%} |"
                    )

        # Detailed results
        lines.append("\n## Detailed Results\n")
        for r in self.results:
            m = r["metrics"]
            avg_score = sum(m.values()) / len(m)
            icon = "✅" if avg_score >= 0.6 else "⚠️" if avg_score >= 0.3 else "❌"
            lines.append(f"### {icon} [{r['category']}] {r['question']}\n")
            lines.append(f"- **Ground Truth:** {r['ground_truth']}")
            lines.append(f"- **Lin Xia:** {r['answer']}")
            lines.append(f"- **Retrieved Contexts:** {len(r['contexts'])}")
            for ctx in r['contexts']:
                lines.append(f"  - `{ctx[:80]}`")
            lines.append(f"- **Scores:** Faithfulness={m['faithfulness']:.2f} | "
                         f"Relevancy={m['answer_relevancy']:.2f} | "
                         f"Precision={m['context_precision']:.2f} | "
                         f"Recall={m['context_recall']:.2f}\n")

        # Technical Details
        lines.append("## Evaluation Methodology\n")
        lines.append("### Metrics")
        lines.append("1. **Faithfulness** — Measures if Lin Xia's response is grounded in retrieved memories (anti-hallucination)")
        lines.append("2. **Answer Relevancy** — Measures if the response actually addresses the question")
        lines.append("3. **Context Precision** — Measures if the top-ranked retrieved memories are the most relevant")
        lines.append("4. **Context Recall** — Measures if all critical memories were retrieved\n")
        lines.append("### Test Categories")
        lines.append("| Category | Count | Description |")
        lines.append("|----------|-------|-------------|")
        lines.append("| fact_recall | 5 | 基本事实记忆召回 |")
        lines.append("| emotional_coherence | 4 | 情感状态的一致性和延续性 |")
        lines.append("| cross_turn | 3 | 跨轮次内容引用 |")
        lines.append("| knowledge_graph | 3 | 知识图谱关联推理 |")
        lines.append("| temporal | 2 | 时间感知 |")
        lines.append("| negation | 2 | 否定与信息纠正 |")
        lines.append("| multi_hop | 2 | 多跳推理 |")
        lines.append("| distractor | 2 | 干扰项鲁棒性 |")
        lines.append("| sensitivity | 2 | 敏感话题边界处理 |")
        lines.append("| relationship | 2 | 关系阶段感知 |")

        report = "\n".join(lines)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📊 Report saved to {output_path}")
        return report

    def save_raw(self, aggregate: dict, output_path: str = "benchmarks/ragas_raw.json"):
        """Save raw evaluation data."""
        data = {
            "aggregate": aggregate,
            "results": self.results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"📦 Raw data saved to {output_path}")

    def cleanup(self):
        """Remove temporary evaluation database."""
        import shutil
        if os.path.exists(self.eval_db_path):
            shutil.rmtree(self.eval_db_path, ignore_errors=True)
            print("🧹 Evaluation temp DB cleaned up")


# ---------------------------------------------------------------------------
# Full RAGAS Integration (optional — requires ragas + API key)
# ---------------------------------------------------------------------------

def run_ragas_full(evaluator: EmotionalRAGEvaluator) -> dict:
    """
    Run full RAGAS evaluation using the ragas library.
    Requires: pip install ragas datasets
    Requires: OPENAI_API_KEY or DEEPSEEK_API_KEY in environment
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        print("[INFO] ragas/datasets not installed. Using built-in heuristic evaluation.")
        print("       To use full RAGAS: pip install ragas datasets")
        return None

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        # Try to use DeepSeek from config
        try:
            from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
            os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
            os.environ["OPENAI_API_BASE"] = DEEPSEEK_BASE_URL
            print("  Using DeepSeek API as LLM-as-a-Judge")
        except ImportError:
            print("[WARNING] No API key available for RAGAS LLM judge.")
            return None

    # Build RAGAS-compatible dataset from our results
    data = {
        "question": [r["question"] for r in evaluator.results],
        "answer": [r["answer"] for r in evaluator.results],
        "contexts": [r["contexts"] for r in evaluator.results],
        "ground_truth": [r["ground_truth"] for r in evaluator.results],
    }
    dataset = Dataset.from_dict(data)

    print("\n  Running RAGAS LLM-as-a-Judge evaluation...")
    try:
        ragas_result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        print(f"  RAGAS Scores: {ragas_result}")
        return dict(ragas_result)
    except Exception as e:
        print(f"  [WARNING] RAGAS evaluation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for Lin Xia")
    parser.add_argument("--category", default=None, help="Evaluate single category")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM generation (infra test only)")
    parser.add_argument("--ragas-full", action="store_true", help="Use full RAGAS with LLM-as-a-Judge")
    args = parser.parse_args()

    # Print testset summary
    s = summary()
    print(f"\n  Testset: {s['total_cases']} cases across {s['category_count']} categories")

    # Run evaluation
    evaluator = EmotionalRAGEvaluator(use_llm_judge=args.ragas_full)

    try:
        aggregate = evaluator.run_full_evaluation(category=args.category)

        if not aggregate:
            print("\n  [ERROR] No results generated.")
            return

        # Print summary
        overall = aggregate.get("overall", {})
        print(f"\n{'='*60}")
        print(f"  EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Faithfulness:      {overall.get('faithfulness', 0):.1%}")
        print(f"  Answer Relevancy:  {overall.get('answer_relevancy', 0):.1%}")
        print(f"  Context Precision: {overall.get('context_precision', 0):.1%}")
        print(f"  Context Recall:    {overall.get('context_recall', 0):.1%}")

        scores = [v for v in overall.values() if isinstance(v, (int, float))]
        composite = sum(scores) / len(scores) if scores else 0
        print(f"\n  🏆 Composite Score: {composite:.1%}")

        # Optionally run full RAGAS
        if args.ragas_full:
            ragas_scores = run_ragas_full(evaluator)
            if ragas_scores:
                aggregate["ragas_llm_judge"] = ragas_scores

        # Save outputs
        evaluator.generate_report(aggregate)
        evaluator.save_raw(aggregate)

    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
