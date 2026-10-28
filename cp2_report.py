"""Readable report construction from computed results; no invented metrics."""
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak)

from cp2_utils import BOUNDS, MEASUREMENTS, SPLINE_FEATURES

DETAIL = Path("output/cp2")


def write_report(meta, table, dictionary, audit, comparison, full, coefficients):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"],
                              alignment=TA_CENTER, fontSize=11, leading=14, spaceAfter=14))
    styles.add(ParagraphStyle(name="Cell", fontSize=7, leading=9, spaceAfter=0))
    styles.add(ParagraphStyle(name="Caption", fontSize=9, leading=12, spaceAfter=10))
    styles["BodyText"].spaceAfter = 8
    styles["BodyText"].leading = 13
    story, markdown = [], []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(escape(text), styles[style]))
        markdown.append(text + "\n")

    def heading(text):
        paragraph(text, "Heading2")

    def formatted_table(frame, caption, widths=None):
        paragraph(caption, "Caption")
        rendered = frame.copy().astype(str)
        rows = [[Paragraph(escape(str(c)), styles["Cell"]) for c in rendered.columns]]
        rows.extend([[Paragraph(escape(value), styles["Cell"]) for value in row] for row in rendered.to_numpy()])
        grid = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        if caption.startswith("Table 2."):
            story[-1].keepWithNext = True
            grid.splitByRow = 0
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcebf0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([grid, Spacer(1, 10)])
        markdown.append(frame.to_csv(index=False) + "\n")

    def figure(filename, caption, max_height=430):
        path = DETAIL / filename
        width, height = ImageReader(str(path)).getSize()
        scale = min(468 / width, max_height / height)
        story.append(Image(str(path), width=width*scale, height=height*scale))
        paragraph(caption, "Caption")

    paragraph("BIOSTAT 707 - Checkpoint 2", "Title")
    paragraph("Wenqing Zhao", "Subtitle")
    heading("AI-use statement")
    paragraph("I used ChatGPT and Codex to help with coding and debugging Python code and generate figures. Codex also helped create the initial skeleton for my PDF report. I gave Codex specific instructions, including keeping my Checkpoint 1 files unchanged. I used ChatGPT to review the code and results, and to turn my ideas for improving figures into instructions for Codex. I also ran commands in the terminal myself to verify the outputs. I wrote my own analysis and interpretation and used Codex to add it to the report. AI was helpful for coding and debugging, but I still had to review its suggestions because they were not always correct or exactly what I wanted.")

    heading("2.1 The modeling table")
    n, deaths = len(table), int(table["In-hospital_death"].sum())
    paragraph("I used Set A, and I also used one row per ICU stay. I used RecordID to match the In-hospital_death outcome from Outcomes-a.txt to each ICU stay. The dataset contains 4,000 stays, and no duplicate IDs were found. There were 554 deaths (13.85%). All the predictors were based on the first 48 hours of ICU measurements.")
    paragraph("I kept the same basic structure from Checkpoint 1, but I changed some cleaning rules based on the problems I found in the data. I treated -1 codes, invalid zeros, and implausible values as missing, I also used fixed ranges for variables such as temperature, pH, and FiO2. And I also kept all 4,000 stays, and handled missing values during model preprocessing.\n\nThese fixed ranges were only used as cleaning assumptions ")

    heading("2.2 Predictors")
    paragraph("I used 262 predictors from the first 48 hours of each ICU stay. And I did not use RecordID, Length_of_stay, or Survival as predictors because they could cause data leakage. SAPS-I and SOFA were only used to compare the models, and I treated ICUType as a categorical variable.")
    paragraph("I used the first, last, minimum, maximum, and mean measurements. I also included measurement counts and missingness indicators because they might tell us more about the patient's condition. I did not remove any variables just because they had a lot of missing values.")
    # Display text only: group membership and counts still come from the saved dictionary.
    predictor_groups = [
        ("Admission", "Admission", 5, "Basic patient information may help predict mortality risk."),
        ("first", "First", 34, "It is the early measurements that may reflect the patient's condition at admission."),
        ("last", "Last", 34, "Most recent measurements may show how the patient's condition changed."),
        ("min", "Min", 34, "Very low values may suggest health problems."),
        ("max", "Max", 34, "Very high values may suggest health problems."),
        ("mean", "Mean", 34, "Average values may give an idea of the patient's overall condition."),
        ("Measurement count", "Measurement count", 35, "Patients who are more seriously ill may need more frequent monitoring."),
        ("Missingness", "Missingness", 49, "Measurement availability / whether a variable was measured"),
        ("Special", "Special", 3, "Urine output, ventilation, and blood pressure may give more information about how well the patient's organs are working."),
    ]
    counts = dictionary.groupby("group").size()
    assert counts.to_dict() == {key: expected for key, _, expected, _ in predictor_groups}
    assert len(counts) == 9 and counts.sum() == len(dictionary) == 262
    groups = pd.DataFrame([(label, counts[key], reason) for key, label, _, reason in predictor_groups],
                          columns=["Predictor group", "Count", "Clinical reason"])
    formatted_table(groups, f"Table 1. Predictor groups used in Models 1 and 3 (262 predictors). The full list of predictors is saved in output/cp2/feature_dictionary.csv.", [110, 45, 313])

    heading("2.3 Cross-validation design")
    paragraph("I used stratified 5-fold cross-validation in this checkpoint 2 and set 7072026 as a random seed.I divided the 4,000 ICU stays into five folds, with each stay appearing in only one fold, and all three models used the same folds. I used a scikit-learn Pipeline to handle missing values and scaling within the training folds to prevent data leakage. I also used 3-fold inner CV to find the best hyperparameters using log loss. I calculated the AUROC for each outer fold and reported the average and sd.")

    heading("2.4 Model 1: ridge logistic regression")
    paragraph("I used ridge logistic regression with 262 predictors. I chose ridge because many predictors were different summaries of the same measurements. The reason that I did not chose LASSO was beacuse it may remove some of these predictors, however, I wanted to keep them all. I used inner CV to choose the penalty strength (C), then refit the model using all of Set A.")
    paragraph("Figure 1 shows the top 20 standardized coefficients because plotting all 267 would be too crowded and hard to read. The full plot is in the appendix. From the Figure 1, it shows that GCS_last had the largest negative coefficient, while Age had the largest positive coefficient. So, from the fiure 1, we can see that higher GCS scores were linked to lower mortality risk which indicates negative relationship, while older age was linked to higher risk which indicates positive relationship.")
    figure("ridge_coefficients_top20.png", "Figure 1. Top 20 standardized ridge coefficients, selected based on their absolute values. Blue bars show positive coefficients, and orange bars show negative coefficients. The full results are in the appendix.", 430)

    heading("2.5 Model 2: restricted cubic spline logistic regression")
    paragraph("II used restricted cubic spline logistic regression with 15 clinical predictors, including 12 continuous variables. I also included 12 missingness indicators to account for missing data. I chose these predictors because I thought they could tell us something about the patient's health and mortality risk. I used inner CV to find the best model settings and kept the same outer CV folds as Model 1. The AUROC results are shown in Table 2.")
    paragraph("Figure 2 shows how Age, HR_mean, and Temp_mean are related to mortality, using odds ratios (ORs) with 95% confidence bands. 65 years, 80 bpm, and 37?C were the reference values here. From the Figure 2, it shows that all three curves have non-linear patterns. From the first figure on the left, as age increases, mortality risk generally increased also, while heart rate and temperature showed approximatly U-shaped patterns. This makes sense because older patients, or those with very low or high heart rates or temperatures, may have a higher risk of death. But, the confidence bands were wider at both ends of the curves, which means the results are less certain in those areas.")
    figure("spline_effects.png", "Figure 2. The effects of Age, HR_mean, and Temp_mean compared with their reference values. The lines show odds ratios (ORs), and the shaded areas show 95% bootstrap confidence bands.", 260)

    heading("2.6 Model 3: random forest")
    paragraph("I used random forest with the same 262 predictors as Model 1. For missing values, I used median imputation. I used 100 trees and tested two options for each setting: maximum depth with None or 10, minimum leaf size with 5 or 20, and maximum features with sqrt or 0.5. I used 3-fold inner CV to choose the best settings using log loss. I did not use early stopping, and I kept the same outer CV folds as Models 1 and 2. The AUROC results are shown in the model comparison table, which are Table 2 and 3.")

    heading("2.7 Comparison and choice of final model")
    paragraph("Tables 2 and 3 show the AUROC results, and Figure 3 shows the ROC curves for all five approaches. From Table 2 and Figure 3, we can see that random forest had the highest average AUROC, which was about 0.846, but ridge logistic regression was very close. Both performed better than SAPS-I and SOFA overall. The differences were also much larger than the variation across folds.")
    display = comparison[["model"] + [f"fold_{f}" for f in range(1, 6)] + ["mean", "sd"]].copy()
    for col in display.columns[1:]:
        display[col] = display[col].map(lambda value: f"{value:.4f}")
    display.columns = ["Approach", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean", "SD"]
    formatted_table(display, "Table 2. AUROC results for the three models, SAPS-I, and SOFA using the same 3,733 ICU stays. The table shows the results from each fold, and also the mean and standard deviation (SD).", [139, 47, 47, 47, 47, 47, 47, 47])
    supplemental = full[["model", "n", "mean", "sd", "oof_log_loss"]].copy()
    for col in ["mean", "sd", "oof_log_loss"]:
        supplemental[col] = supplemental[col].map(lambda value: f"{value:.4f}")
    formatted_table(supplemental, "Table 3. AUROC results for the three models using all 4,000 ICU stays. The table shows the mean, SD, and out-of-fold log loss.", [210, 45, 65, 65, 83])
    best = comparison.loc[comparison.key.isin(["ridge", "spline", "forest"])].sort_values("mean", ascending=False).iloc[0]
    folds = [f"fold_{i}" for i in range(1, 6)]
    for benchmark in ["SAPS-I", "SOFA"]:
        score = comparison.loc[comparison.key.eq(benchmark)].iloc[0]
        differences = best[folds].to_numpy(float) - score[folds].to_numpy(float)
        spread = np.std(differences, ddof=1)
        conclusion = "exceeds one SD of the paired fold differences" if differences.mean() > spread else "does not exceed one SD of the paired fold differences"
    figure("roc_comparison.png", "Figure 3. ROC curves for all five approaches. The AUROC values shown in Figure 3 use all out-of-fold predictions, so they may be a little different from the averages in Table 2.", 400)
    paragraph("I chose ridge logistic regression because it performed almost as well as random forest, but it was easier to understand. I then trained the final model again using all 4,000 ICU stays in Set A and saved it along with the out-of-fold predictions in the output folder for Checkpoint 3. And from Figure 3, we can see that ridge logistic regression and random forest had similar ROC curves with the same AUROC number.")

    heading("Limitations")
    story[-1].keepWithNext = True
    paragraph("For this checkpoint 2, I only used Set A and all the data from the first 48 hours of each ICU stay. Some values were missing when doing this checkpoint, and the cleaning rules might not work well for every patient. Also, these results only show how the predictors are related to mortality risk. They do not mean that one directly causes the other.")
    if (DETAIL / "ridge_coefficients_panel_1.png").exists():
        for panel in range(1, 5):
            story.append(PageBreak())
            if panel == 1:
                heading("Appendix: complete ridge coefficients")
            count = 67 if panel < 4 else 66
            figure(f"ridge_coefficients_panel_{panel}.png", ("Figure A1a. The first 67 standardized ridge coefficients.", "Figure A1b. The next 67 standardized ridge coefficients.", "Figure A1c. Another 67 standardized ridge coefficients.", "Figure A1d. The last 66 standardized ridge coefficients.")[panel - 1], 565 if panel == 1 else 600)

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(54, 26, "BIOSTAT 707 | Wenqing Zhao | Checkpoint 2 Report")
        canvas.drawRightString(558, 26, str(doc.page))

    doc = SimpleDocTemplate("output/checkpoint2.pdf", pagesize=(612, 792),
                            leftMargin=54, rightMargin=54, topMargin=45, bottomMargin=45,
                            title="BIOSTAT 707 Checkpoint 2", author="Wenqing Zhao")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (DETAIL / "checkpoint2_report.md").write_text("\n".join(markdown), encoding="utf-8")
