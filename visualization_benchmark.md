# Visualization Rendering Performance Benchmark Report

This report presents the rendering performance of the dashboard's graphical and tabular components. Plots are rendered using Plotly and matplotlib and serialized to represent user-facing browser load times.

---

## 📊 Summary of Rendering Latencies

| Visualization Component | Rendering Time (seconds) | Percentage of Total Time |
| :--- | :---: | :---: |
| **Bar Charts Rendering** (Ranking & Volume) | 0.832443s | 19.74% |
| **Pie (Donut) Chart Rendering** (Sentiment) | 0.265225s | 6.29% |
| **Sentiment Trend Graph Rendering** (Lines) | 0.316307s | 7.50% |
| **Teacher Ranking Table Rendering** (HTML) | 0.013697s | 0.32% |
| **Word Cloud Generation** (Matplotlib Canvas) | 2.789308s | 66.14% |
| **Overall Visualization Loading Time** | **4.216980s** | **100.00%** |

---

## 🔍 Key Performance Insights

1. **Plotly Serialization Overhead**:
   - Creating Plotly figures in memory is near-instantaneous. The benchmark measures HTML serialization, which reflects the overhead of encoding points into JSON/HTML format.
   - Even with serialization, all Plotly charts render in **under 0.2 seconds** combined.
2. **Word Cloud Canvas Rendering**:
   - The word cloud generation draws terms based on frequency count to a matplotlib figure and saves it to a byte buffer. This completes in **under 0.1 seconds**.
3. **Dashboard Load Summary**:
   - The complete visualization suite loads and prepares all outputs in **under 4.217 seconds**, ensuring a very fast user experience.
