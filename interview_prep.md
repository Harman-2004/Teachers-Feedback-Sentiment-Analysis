# Technical Interview Preparation Guide: Teacher Feedback Analytics

This guide is designed for preparing for technical interviews, focusing on the engineering decisions, performance optimizations, and architectural patterns of the Teacher Feedback Analytics project.

---

## 🐍 Section 1: Python

### Q1.1: Why are libraries like `PyTorch` or `transformers` imported lazily inside functions rather than at the module level?
*   **Difficulty**: Mid
*   **Ideal Answer**: 
    Importing heavy libraries like `torch` or `transformers` at the module level forces the Python interpreter to load them immediately when the file is imported. Since these libraries take several seconds to load, this degrades CLI command start times, slows down dashboard cold starts, and makes testing slower. Importing them lazily inside functions (e.g. `_load_pipeline`) ensures that they are only loaded if and when a user explicitly initiates deep learning analysis.
*   **Explanation**: 
    In Python, `import` statements execute code that builds module objects and registers them in `sys.modules`. Lazy imports defer this overhead. In this project, running with `FORCE_RULE_BASED=1` skips deep learning models entirely. If imports were at the module level, PyTorch would still load and potentially crash or consume system memory.
*   **Follow-up Questions**:
    1. How would you programmatically verify that a library hasn't been loaded in `sys.modules`?
    2. What are the downsides of lazy imports (e.g., hiding syntax/import errors until runtime)?

### Q1.2: How does `functools.lru_cache` work, and why is it useful for model loading?
*   **Difficulty**: Mid
*   **Ideal Answer**: 
    `lru_cache` (Least Recently Used cache) is a decorator that memorizes a function's returns based on its arguments. For model loading (e.g., `_load_encoder` or `_load_summarizer`), using `@lru_cache(maxsize=1)` ensures that the model is loaded from disk/memory into RAM exactly once. Subsequent calls with the same model name return the cached object instantaneously.
*   **Explanation**: 
    Loading deep learning weights involves heavy I/O operations and memory allocation. Without caching, calling `detect_aspects` or `generate_summary` on different teachers would reload the model weights on every single call, causing severe latency and memory leaks.
*   **Follow-up Questions**:
    1. What happens if the cached model object is mutated in-place?
    2. How does `lru_cache` determine cache key hits (argument equality and hashing)?

---

## 🐼 Section 2: Pandas & CSV Processing

### Q2.1: How do you handle missing values (NaNs) in dates and numeric ratings during data ingestion without crashing the pipeline?
*   **Difficulty**: Mid
*   **Ideal Answer**: 
    We use Pandas' parsing utilities with error-coercion. For dates, `pd.to_datetime(df['date'], errors='coerce')` converts invalid dates or missing entries to `NaT` (Not a Time). For ratings, `pd.to_numeric(df['rating'], errors='coerce')` converts non-numeric entries to `NaN`. We then fill missing ratings with the column's mean/median or drop rows where critical columns (like `feedback_text`) are missing.
*   **Explanation**: 
    If you do not use `errors='coerce'`, Pandas will raise a `ValueError` if a single row has an invalid string format. Coercing turns those fields into null elements, allowing downstream filtering via `.dropna()` or imputation via `.fillna()`.
*   **Follow-up Questions**:
    1. How do timezone differences affect `pd.to_datetime`?
    2. When would you prefer median imputation over mean imputation for ratings?

### Q2.2: How would you optimize group-by aggregation and ranking operations in Pandas for a dataset exceeding 100,000 feedback comments?
*   **Difficulty**: Senior
*   **Ideal Answer**: 
    First, we filter out irrelevant columns early to minimize memory footprint. Second, we ensure grouping columns (like `teacher_name`) use the `category` datatype rather than `object` (string). Third, we perform aggregation in a single pass using `.groupby().agg()`. Finally, we calculate ranks using Pandas' vectorised `.rank(ascending=False, method='min')` rather than sorting or looping in Python.
*   **Explanation**: 
    Categorical datatypes represent strings as integer codes under the hood, making grouping operations up to 10x faster and using significantly less memory. Vectorized operations run in compiled C/C++ libraries (under NumPy/Pandas) rather than slow Python loops.
*   **Follow-up Questions**:
    1. What is the difference between `method='min'` and `method='dense'` in ranking?
    2. How would you handle out-of-core data processing if the dataset exceeds system RAM (e.g., using Dask or Polars)?

---

## 🧠 Section 3: NLP & Sentiment Analysis

### Q3.1: How does a rule-based sentiment analyzer work, and how does it compare to TextBlob?
*   **Difficulty**: Mid
*   **Ideal Answer**: 
    A rule-based analyzer tokenizes text, matches words against a pre-defined lexicon of positive and negative words, and adjusts scores based on modifiers (like negations and intensifiers). 
    `TextBlob` uses a similar pattern-based lexicon approach under the hood, calculating a polarity score between -1 and +1 and subjectivity between 0 and 1. Our rule-based implementation is tailored for education contexts (e.g., treating "workload" or "monotonous" as negative indicators).
*   **Explanation**: 
    Lexicon methods check token containment. If a positive word is preceded by a negation word (e.g., "not good"), the polarity is inverted. If preceded by an intensifier (e.g., "very good"), the score is multiplied.
*   **Follow-up Questions**:
    1. What are the main limitations of lexicon-based sentiment analysis (e.g., sarcasm, domain-specific meanings)?
    2. How does TextBlob handle words that can be both nouns and adjectives?

### Q3.2: Explain the difference between extractive and abstractive summarization. Which one is used as a fallback and why?
*   **Difficulty**: Mid
*   **Ideal Answer**: 
    **Extractive summarization** ranks existing sentences from the text based on word importance (e.g., TF-IDF or text rank) and pulls the top sentences verbatim. 
    **Abstractive summarization** uses a generative model (like BART or T5) to write entirely new sentences that synthesize the text.
    In this project, extractive summarization is the fallback because it requires zero deep learning weights, runs instantly using rule-based scoring, and does not require GPU/high-memory availability.
*   **Explanation**: 
    Generative models require loading billions of parameters, which is prone to memory crashes on CPU-bound dashboard hosting servers. Extractive methods use token intersection formulas that are lightweight and mathematically guaranteed not to hallucinate.
*   **Follow-up Questions**:
    1. How do you evaluate the quality of abstractive vs extractive summaries (e.g., ROUGE scores)?
    2. What are the risks of hallucination in abstractive summarization?

---

## 🤖 Section 4: Transformers & ABSA

### Q4.1: How does Aspect-Based Sentiment Analysis (ABSA) differ from document-level sentiment analysis?
*   **Difficulty**: Senior
*   **Ideal Answer**: 
    Document-level sentiment analysis assigns a single overall polarity score to an entire text. ABSA breaks the text down into specific aspects (e.g., "Communication", "Assignment Quality") and calculates a separate sentiment score for each aspect. A single feedback comment like "The teacher is very clear but the assignments are too hard" is neutral overall, but ABSA yields positive for *Communication* and negative for *Assignment Quality*.
*   **Explanation**: 
    ABSA is typically implemented in two stages: Aspect Extraction (identifying which aspects are discussed) and Aspect Sentiment Classification (determining the polarity towards those aspects).
*   **Follow-up Questions**:
    1. How does your model handle overlapping aspects in a single sentence?
    2. What is the difference between pipeline ABSA and joint/end-to-end ABSA?

### Q4.2: How would you implement zero-shot aspect classification using SentenceTransformers (embeddings)?
*   **Difficulty**: Senior
*   **Ideal Answer**: 
    We define anchor sentences for each aspect representing positive and negative sentiments (e.g., "The explanation was clear" vs "The speed was too fast"). We then use a SentenceTransformer model to embed both the input feedback sentences and the anchor sentences into a high-dimensional vector space. By calculating the Cosine Similarity between the input sentence vector and the anchor vectors, we classify the sentence into the aspect that yields the highest similarity score.
*   **Explanation**: 
    Cosine similarity is defined as:
    \[\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}\]
    This measures the alignment of vectors in space, representing semantic similarity regardless of literal keyword matches.
*   **Follow-up Questions**:
    1. How does the choice of pooling (mean vs cls) affect sentence embeddings?
    2. Why does Cosine Similarity work better than Euclidean Distance for high-dimensional embeddings?

---

## 🖥️ Section 5: Streamlit & Plotly

### Q5.1: How does Streamlit's execution model work, and why does it necessitate state management?
*   **Difficulty**: Mid
*   **Ideal Answer**: 
    Streamlit runs the entire Python script from top to bottom every time a user interacts with a widget (clicks a button, changes a slider). Because of this, variables are reset on every rerun. To persist data across reruns (such as loaded datasets, model outputs, or active page tabs), we must store them in `st.session_state` (a key-value dictionary that persists across user sessions).
*   **Explanation**: 
    This top-down execution model simplifies dashboard creation but requires developer discipline to prevent redundant computations (like reading databases or running inference) by using `st.cache_data`, `st.cache_resource`, or `st.session_state`.
*   **Follow-up Questions**:
    1. What is the difference between `st.cache_data` and `st.cache_resource`?
    2. What happens to session state when a user closes or refreshes their browser tab?

### Q5.2: What is Plotly figure serialization, and how does it affect rendering latency?
*   **Difficulty**: Senior
*   **Ideal Answer**: 
    Plotly figures are represented as Python dictionary trees. To render them in a browser, Streamlit must serialize these Python objects to JSON (or export them to HTML) and send them to the frontend where the Plotly.js library renders them using WebGL/SVG. 
    Serialization latency becomes a bottleneck when there are thousands of data points. We optimize this by downsampling data points before plotting and disabling inline JS packages in the export string (using `include_plotlyjs=False`).
*   **Explanation**: 
    Converting complex figures to text (HTML/JSON strings) is CPU-intensive. Caching the serialized strings (or generated figure objects) or rendering them client-side preserves dashboard responsiveness.
*   **Follow-up Questions**:
    1. How would you embed Plotly charts dynamically inside custom HTML components?
    2. When would you choose Plotly over Matplotlib/Seaborn for dashboard visualization?
