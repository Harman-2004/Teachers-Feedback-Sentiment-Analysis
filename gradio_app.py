"""
gradio_app.py
-------------
Gradio dashboard application for Teacher Feedback Sentiment Analysis.
Uses the trained winning ML model pipeline (models/sentiment_model.joblib).

Run:
    python gradio_app.py
"""

from __future__ import annotations
import os
import gradio as gr
import pandas as pd

from nlp.sentiment import analyze_sentiment_ml
from nlp.preprocessor import preprocess_text


def predict_sentiment_gradio(feedback_text: str):
    """Gradio prediction endpoint for a single teacher feedback string."""
    if not feedback_text or not feedback_text.strip():
        return "Empty Input", 0.0, "5.0 / 10.0", "Please enter valid feedback text."

    results = analyze_sentiment_ml([feedback_text])
    res = results[0]

    label = res.get("label", "Neutral")
    confidence = res.get("confidence", 0.0)
    polarity = res.get("polarity", 5.0)
    cleaned = preprocess_text(feedback_text)

    summary_msg = (
        f"**Predicted Sentiment**: `{label}`\n\n"
        f"**Model Confidence**: `{confidence * 100:.1f}%`\n\n"
        f"**Polarity Score**: `{polarity:.1f} / 10.0`\n\n"
        f"**Preprocessed Tokens**: `{cleaned}`"
    )

    return label, float(confidence), f"{polarity:.1f} / 10.0", summary_msg


def predict_batch_gradio(batch_text: str):
    """Gradio prediction endpoint for multiple feedback comments (one per line)."""
    if not batch_text or not batch_text.strip():
        return pd.DataFrame(columns=["Feedback Text", "Predicted Sentiment", "Polarity Score"])

    lines = [l.strip() for l in batch_text.strip().splitlines() if l.strip()]
    results = analyze_sentiment_ml(lines)

    rows = []
    for r in results:
        rows.append({
            "Feedback Text": r["text"],
            "Predicted Sentiment": r["label"],
            "Polarity Score": f"{r['polarity']:.1f} / 10.0"
        })
    return pd.DataFrame(rows)


# Build Gradio Blocks Dashboard
with gr.Blocks(title="Teacher Feedback Sentiment Analysis") as demo:
    gr.Markdown(
        """
        # 🎓 Teacher Feedback Sentiment Analysis Dashboard
        ### Powered by ML (Word2Vec + Random Forest Pipeline)
        Enter student feedback comments below to analyze sentiment polarity, classification confidence, and preprocessed tokens.
        """
    )

    with gr.Tab("Single Feedback Analysis"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    lines=4,
                    placeholder="Enter student feedback comment here...",
                    label="Teacher Feedback Text"
                )
                analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
            with gr.Column():
                out_label = gr.Textbox(label="Predicted Sentiment Class")
                out_conf = gr.Number(label="Confidence Score")
                out_pol = gr.Textbox(label="Polarity Score (0-10)")
                out_details = gr.Markdown(label="Analysis Details")

        analyze_btn.click(
            fn=predict_sentiment_gradio,
            inputs=[input_text],
            outputs=[out_label, out_conf, out_pol, out_details]
        )

        gr.Examples(
            examples=[
                ["Creates an incredibly interactive learning environment where every student feels heard."],
                ["Feedback on assignments is generic, copy-pasted, and returns no useful guidance."],
                ["The engagement level is average — some days are interactive, but most follow standard lecture format."],
                ["Explains historical context well, but graded assignments are returned weeks after deadline."],
                ["Great teacher."],
                [""]
            ],
            inputs=[input_text]
        )

    with gr.Tab("Batch Feedback Analysis"):
        batch_input = gr.Textbox(
            lines=6,
            placeholder="Enter multiple feedback comments (one per line)...",
            label="Batch Teacher Feedback Input"
        )
        batch_btn = gr.Button("📊 Run Batch Inference", variant="primary")
        batch_output = gr.Dataframe(label="Batch Sentiment Classification Results")

        batch_btn.click(
            fn=predict_batch_gradio,
            inputs=[batch_input],
            outputs=[batch_output]
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
