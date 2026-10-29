# SEval-Ex: Statement-Level Framework for Explainable Summarization Evaluation

SEval-Ex is a framework that bridges the gap between performance and interpretability in summarization evaluation. It decomposes summarization evaluation into atomic statements, providing both high correlation with human judgment and explainable assessment of factual consistency.

## 📋 Overview

SEval-Ex employs a two-stage pipeline:
1. **Statement Extraction**: Uses LLMs to decompose both source and summary texts into atomic statements
2. **Verdict Reasoning**: Matches and classifies statements as True Positives (TP), False Positives (FP), or False Negatives (FN)

Unlike existing approaches that provide only summary-level scores, SEval-Ex generates detailed evidence for its decisions through statement-level alignments.

## 🔧 Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) for local LLM access

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/TanguyHsrt/seval-ex
   cd seval-ex
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running with the required models:
   ```bash
   ollama pull qwen2.5:72b
   ```

## 🚀 Quick Start

### Basic Usage: Evaluate a Single Summary

```python
from seval_package import SEvalPipeline

# Initialize the pipeline
evaluator = SEvalPipeline(model_name="qwen2.5:72b", method="StSum_Text")

# Evaluate a single summary
source_text = "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity and won the Nobel Prize in Physics in 1921."
summary = "Einstein was born in Germany and won a Nobel Prize."

result = evaluator.evaluate_single(source_text, summary)

# Print the metrics
print(f"Precision: {result['metrics']['precision']:.4f}")
print(f"True Positives: {result['metrics']['tp_count']}")
print(f"False Positives: {result['metrics']['fp_count']}")
```

### Command Line Interface

The package also provides a command-line interface:

```bash
python seval_package.py --text "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity and won the Nobel Prize in Physics in 1921." --summary "Einstein was born in Germany and won a Nobel Prize." --model "qwen2.5:72b"
```

### Evaluating a Dataset

```python
from seval_package import SEvalPipeline
from datasets import load_dataset

# Load SummEval dataset
dataset = load_dataset("mteb/summeval")["test"]

# Initialize the pipeline
evaluator = SEvalPipeline(model_name="qwen2.5:72b")

# Evaluate multiple examples
results = evaluator.evaluate_dataset(dataset)

# Display aggregated metrics
print(f"Average F1: {results['aggregated_metrics']['f1']:.4f}")
print(f"Average Precision: {results['aggregated_metrics']['precision']:.4f}")
print(f"Average Recall: {results['aggregated_metrics']['recall']:.4f}")
```

## 🧩 Pipeline Variants

SEval-Ex supports different evaluation methods to handle various text lengths and complexity:

- **StSum_Text**: Direct matching of summary statements to source text (default, best performance)
- **Base**: Processing the entire text as a single unit
- **Chunked**: Segmenting the text into smaller chunks for better context preservation

```python
# Choose a pipeline variant
base_evaluator = SEvalPipeline(method="Base")
chunked_evaluator = SEvalPipeline(method="Chunked", chunk_size=3)
st_sum_evaluator = SEvalPipeline(method="StSum_Text")  # Default

# Compare results
base_result = base_evaluator.evaluate_single(source_text, summary)
chunked_result = chunked_evaluator.evaluate_single(source_text, summary)
st_sum_result = st_sum_evaluator.evaluate_single(source_text, summary)

print(f"Base F1: {base_result['metrics']['f1']:.4f}")
print(f"Chunked F1: {chunked_result['metrics']['f1']:.4f}")
print(f"StSum_Text F1: {st_sum_result['metrics']['f1']:.4f}")
```

## 🔍 Hallucination Detection

SEval-Ex includes a module for generating and evaluating hallucinated summaries of three types:

1. **Entity Replacement**: Replacing named entities with incorrect ones
2. **Incorrect Events**: Modifying the sequence of events
3. **Fictitious Details**: Adding plausible but unsupported details

```python
from hallucination_evaluator import HallucinationEvaluator, HallucinationType
from seval_package import SEvalPipeline

# Initialize evaluators
hallucination_evaluator = HallucinationEvaluator(model_name="qwen2.5:72b")
seval_evaluator = SEvalPipeline()

# Generate hallucinated versions of a summary
source_text = "The company announced a new product line yesterday during their annual conference. CEO Jane Smith presented the features and pricing details."
original_summary = "The company announced a new product line at their annual conference, presented by the CEO."

# Generate hallucinated summaries
entity_hallucination = hallucination_evaluator.generate_hallucination(
    original_summary, 
    HallucinationType.ENTITY_REPLACEMENT
)
event_hallucination = hallucination_evaluator.generate_hallucination(
    original_summary, 
    HallucinationType.INCORRECT_EVENT
)
detail_hallucination = hallucination_evaluator.generate_hallucination(
    original_summary, 
    HallucinationType.FICTITIOUS_DETAILS
)

# Evaluate the original and hallucinated summaries
original_result = seval_evaluator.evaluate_single(source_text, original_summary)
entity_result = seval_evaluator.evaluate_single(source_text, entity_hallucination)
event_result = seval_evaluator.evaluate_single(source_text, event_hallucination)
detail_result = seval_evaluator.evaluate_single(source_text, detail_hallucination)

# Compare the scores
print(f"Original F1: {original_result['metrics']['f1']:.4f}")
print(f"Entity Replacement F1: {entity_result['metrics']['f1']:.4f}")
print(f"Incorrect Event F1: {event_result['metrics']['f1']:.4f}")
print(f"Fictitious Details F1: {detail_result['metrics']['f1']:.4f}")
```

## 📊 Expected Outputs

When running the SEval-Ex pipeline, expect output like this:

```
Evaluation Results:
Precision: 0.750
Recall: 0.667
F1: 0.706
True Positives: 3
False Positives: 1
False Negatives: 1

Extracted Statements:
- Einstein was born in Germany.
- Einstein won a Nobel Prize.

Verdict Reasoning:
True Positives:
+ Einstein was born in Germany. VERDICT: TP
+ Einstein won a Nobel Prize. VERDICT: TP

False Positives:
- None

False Negatives:
- Einstein was born in 1879. VERDICT: FN
- Einstein developed the theory of relativity. VERDICT: FN
```

## 🔌 API Reference

### SEvalPipeline

The main class for running summarization evaluations.

```python
SEvalPipeline(
    model_name: str = "qwen2.5:72b",
    method: str = "StSum_Text",
    chunk_size: int = 3,
    max_text_length: int = 500
)
```

**Methods**:
- `evaluate_single(text: str, summary: str) -> Dict`: Evaluate a single summary
- `evaluate_dataset(dataset, text_key: str = "text", summary_key: str = "summary") -> Dict`: Evaluate all examples in a dataset
- `extract_statements(text: str) -> List[str]`: Extract atomic statements from text
- `verify_statements(summary_statements: List[str], text: str) -> Dict[str, List[str]]`: Verify summary statements against source text

### HallucinationEvaluator

Class for generating and evaluating hallucinated summaries.

```python
HallucinationEvaluator(
    model_name: str = "qwen2.5:72b",
    temperature: float = 0.0001
)
```

**Methods**:
- `generate_hallucination(summary: str, hallucination_type: HallucinationType) -> str`: Generate a hallucinated version of the summary
- `evaluate_with_seval(hallucinated_summary: str, source_text: str, seval_pipeline) -> Dict`: Evaluate hallucinated summary using SEval pipeline
- `evaluate_summeval_dataset(dataset_name: str = "mteb/summeval", seval_pipeline = None, sample_size: Optional[int] = None) -> Dict`: Evaluate hallucinations on the SummEval dataset

## 📚 Citation

If you use SEval-Ex in your research, please cite the original paper:

```
(Coming soon)
```

## 📜 License

This project is licensed under the MIT License.
