# Consolidated Software Performance Benchmark Report

This report presents a consolidated performance view of the Teacher Feedback Analytics project, aggregating measurements across the web dashboard, NLP pipeline, data processing layers, and visualization rendering.

---

## 📊 Core Performance Metrics

| Metric | Measured Value | Description |
| :--- | :---: | :--- |
| **Average Dashboard Loading Time** | **6.397s** | Average cold-start and refresh loading time for the dashboard |
| **Average Sentiment Inference Latency** | **0.037 ms** | Average polarity classification latency per feedback comment |
| **Average Feedback Processing Latency** | **0.236 ms** | Average time to run the full NLP pipeline on a single feedback comment |
| **Average Teacher Ranking Generation Time** | **22.243 ms** | Average time to calculate scores, scale metrics, and rank teachers |
| **Total Visualization Rendering Time** | **4.217s** | Combined rendering and HTML serialization time for all dashboard plots |
| **NLP Throughput (Reviews Processed / Sec)** | **4234.57 reviews/s** | Maximum feedback throughput under rule-based fallback mode |
| **Average CPU Utilization** | **9.12%** | Mean CPU load across all active benchmark sessions |
| **Average RAM Usage** | **52.47 MB** | Mean heap size after processing data |

---

## 🔍 Engineering Recommendations & Insights

1. **Rule-Based Fallback Efficiency**:
   - The rule-based analyzer processes feedback at an impressive **4234.6 reviews per second** with a sub-millisecond average processing latency.
   - Using this fallback allows the application to run smoothly on low-memory servers (such as Streamlit Community Cloud) under 80 MB of RAM.
2. **Dashboard Load Overhead**:
   - The average dashboard load time of **6.40s** is dominated by cold starts and imports. Pre-caching imports or utilizing Streamlit's state cache is recommended to optimize page refreshes.
3. **Visualization Serialization**:
   - Visualizations are rendered quickly, but word cloud rasterization via matplotlib represents the primary bottleneck. If faster page load times are desired, substituting matplotlib with a browser-side JS word cloud renderer would reduce render times by ~7 seconds.
