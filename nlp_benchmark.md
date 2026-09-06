# NLP Pipeline Performance Benchmark Report

This report presents the performance of the core Natural Language Processing (NLP) components of the Teacher Feedback Analytics project, evaluated on a dataset of exactly 500 student feedback comments.

---

## 📊 Summary of NLP Performance Metrics

| Pipeline Phase | Time Spent (seconds) | Percentage of Total Time |
| :--- | :---: | :---: |
| **Text Preprocessing** | 0.002727s | 2.31% |
| **Tokenization** | 0.010416s | 8.82% |
| **Stopword Removal** | 0.002060s | 1.74% |
| **Lemmatization** | 0.008323s | 7.05% |
| **Sentiment Inference** | 0.018701s | 15.84% |
| **Aspect-Based Sentiment Analysis (ABSA)** | 0.041304s | 34.98% |
| **Feedback Summarization** | 0.034547s | 29.26% |
| **Total NLP Pipeline Execution** | **0.118076s** | **100.00%** |

---

## 📈 Processing Throughput & Latency

*   **Total Reviews Processed**: 500
*   **Reviews Processed Per Second**: **4234.57 reviews/sec**
*   **Average Inference Latency**: **0.236 ms / review**

---

## 🔍 Key Performance Insights

1.  **Rule-Based Pipeline Speed**:
    *   Running the NLP pipeline under the optimized rule-based fallback mode completes the processing of 500 reviews in **under 0.118 seconds**.
    *   Throughput averages **4234.6 reviews per second**, showing the high efficiency of local keyword and suffix-stripping lookups.
2.  **Bottleneck Analysis**:
    *   The most intensive step in rule-based mode is aspect-based sentiment analysis, which matches keywords for five dimensions across sentences.
    *   Tokenization, stopword removal, and lemmatization account for a minor percentage of overall latency.
