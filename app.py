import gradio as gr
import joblib
import numpy as np
import pandas as pd

# ==== Load model & artefak ====
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
le_nat = joblib.load("le_nationality.pkl")
le_edu = joblib.load("le_education.pkl")
feature_cols = joblib.load("feature_cols.pkl")

NATIONALITY_OPTIONS = list(le_nat.classes_)
EDUCATION_OPTIONS = list(le_edu.classes_)


def predict_kiip_level(model_choice, age, nationality, study_hours, reading, listening,
                        writing, speaking, grammar, vocabulary, mock_test,
                        education, topik_level, years_in_korea):

    nationality_enc = le_nat.transform([nationality])[0]
    education_enc = le_edu.transform([education])[0]

    input_df = pd.DataFrame([{
        "Age": age,
        "Nationality_enc": nationality_enc,
        "Study_Hours": study_hours,
        "Reading": reading,
        "Listening": listening,
        "Writing": writing,
        "Speaking": speaking,
        "Grammar": grammar,
        "Vocabulary": vocabulary,
        "Mock_Test": mock_test,
        "Education_enc": education_enc,
        "TOPIK_Level": topik_level,
        "Years_in_Korea": years_in_korea,
    }])[feature_cols]

    if model_choice == "Random Forest":
        pred = rf_model.predict(input_df)[0]
        proba = rf_model.predict_proba(input_df)[0]
    else:
        pred = xgb_model.predict(input_df)[0]
        proba = xgb_model.predict_proba(input_df)[0]

    proba_dict = {f"Level {i}": float(p) for i, p in enumerate(proba)}

    return f"Prediksi KIIP Level: {int(pred)}", proba_dict


with gr.Blocks(title="Prediksi Level KIIP") as demo:
    gr.Markdown("""
    # 🇰🇷 Sistem Prediksi Penempatan Level KIIP
    Prediksi level penempatan **Korean Immigration and Integration Program (KIIP)**
    berdasarkan kemampuan bahasa Korea, menggunakan model **Random Forest** atau **XGBoost**.
    """)

    with gr.Row():
        with gr.Column():
            model_choice = gr.Radio(["Random Forest", "XGBoost"], value="Random Forest", label="Pilih Model")
            age = gr.Slider(15, 70, value=30, step=1, label="Usia (Age)")
            nationality = gr.Dropdown(NATIONALITY_OPTIONS, value=NATIONALITY_OPTIONS[0], label="Kewarganegaraan")
            education = gr.Dropdown(EDUCATION_OPTIONS, value=EDUCATION_OPTIONS[0], label="Pendidikan Terakhir")
            study_hours = gr.Slider(0, 900, value=400, step=1, label="Total Jam Belajar (Study Hours)")
            years_in_korea = gr.Slider(0, 10, value=2, step=0.1, label="Lama Tinggal di Korea (Tahun)")
            topik_level = gr.Slider(0, 6, value=3, step=1, label="Level TOPIK")

        with gr.Column():
            reading = gr.Slider(0, 100, value=70, step=1, label="Reading")
            listening = gr.Slider(0, 100, value=70, step=1, label="Listening")
            writing = gr.Slider(0, 100, value=70, step=1, label="Writing")
            speaking = gr.Slider(0, 100, value=70, step=1, label="Speaking")
            grammar = gr.Slider(0, 100, value=70, step=1, label="Grammar")
            vocabulary = gr.Slider(0, 100, value=70, step=1, label="Vocabulary")
            mock_test = gr.Slider(0, 110, value=70, step=1, label="Skor Mock Test")

    predict_btn = gr.Button("Prediksi Level KIIP", variant="primary")

    output_label = gr.Textbox(label="Hasil Prediksi")
    output_proba = gr.Label(label="Probabilitas Tiap Level")

    predict_btn.click(
        fn=predict_kiip_level,
        inputs=[model_choice, age, nationality, study_hours, reading, listening,
                writing, speaking, grammar, vocabulary, mock_test,
                education, topik_level, years_in_korea],
        outputs=[output_label, output_proba]
    )

if __name__ == "__main__":
    demo.launch()
