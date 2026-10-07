# Checkpoint 1 Writeup
# Vicky Zhao
## Table 1 and Outcome Summary

# 1. What information exists at prediction time (the 48-hour window).
During the 48-hour window of the prediction time, we have the patient information and measurements collected within the first 48 hours of the ICU stay. I summarized the repeated measurements over the full 48 hours using count, first, last, minimum, maximum, and mean in the set_a_wide.csv. Some measurements are not available for every patient and are treated as missing, while the in-hospital death outcome would not yet be known and therefore is not used as a predictor.

# 2. Table 1 and outcome summary, with two or three sentences each.
### Table 1
From the table 1, patients who died had higher average age than patients who survived, with a mean age of 69.97 years compared with 63.33 years. And mean height was similar between two groups, but patients who died had slightly higher average weight than patients who did not. 

### Outcome Summary
There were 4,000 participants in Set a. Throughout all the 4000 participants, there are 554 patients (13.85%) died in the hospital and 3,446 (86.15%) did not.

# 3. Missingness: which variables, for whom, and whether it is plausibly informative. Reference the map and the missingness-versus-death table.
Missingness was different across the variables. From output/missingness_map.png, TroponinI, Cholesterol, and TroponinT were missing for many patients, while HR, Creatinine, GCS, and BUN were available for most patients. Missingness also appeared to be related to patient outcome. For example, the death rate was 19.1% when ALP was measured, compared with 10.0% when it was missing. Lactate showed a similar pattern. This may mean that some tests were more likely to be ordered for sicker patients.

# 4. AI-use statement: tool, what for, how verified.
I used ChatGPT to help me understand parts of the assignment, explain some of the Python code and the provided skeleton, and troubleshoot errors while I was working. I also used it to help me understand how methods shown in the class slides could be applied to the Set A data and to clarify the general steps of the analysis when I was unsure what to do next. I verified the suggestions by running the code myself, checking the output files, tables, and figures.