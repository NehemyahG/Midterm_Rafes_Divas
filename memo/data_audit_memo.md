=== Missing Value ===

Missing value are less than 15 for all features, neglectable.

=== Skewness Value ===
Skewness value for all features are less than 0.5 (biggest glucose_mean (Skewness: 0.023412652049166805))

=== Correlation values by moratality_label ===
There is no significant correlation between features and mortality_label

=== Cohen's d ====
Cohen's d shows small effect size for all features between mortality_label == 0 and mortality _label == 1.
Cohen's d => 0.14 was observed => apache_score, sofa_score, comorbidity_score, and age.

It suggests that individual features do not strongly separate mortality groups on their own. This likely indicates that mortality prediction in this dataset depends on multiple interacting clinical variables rather than single dominant predictors, which is common in complex medical datasets.


=== Feature Enginnering ===
shock_index -> Predictor of mortality and sepsis
heart_rate_range -> Capture instability in heart rate, which can be a sign of critical illness
urine_output_daily -> Normalize urine output by length of stay to account for differences in monitoring duration
severity_combined -> Combine SOFA and APACHE II scores for overall severity assessment
critical_support -> Flag for patients requiring critical support
resp_oxygen_ratio -> Calculate the ratio for assessing respiratory function


=== Class Imbalance ===
Class ration (1/0) == ~30% (moderate imbalance). 

The dataset exhibits moderate class imbalance, with approximately 30% mortality cases and 70% survival cases. To address this imbalance, stratified sampling was used during train–test splitting and evaluation metrics such as ROC-AUC and Precision–Recall AUC were prioritized. Additionally, class weighting was applied during model training to ensure that the minority class received appropriate attention during learning.

How to deal with this?
* Use stratified split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        stratify=y,
        random_state=42
    )
* Use ROC-AUC and PR-AUC
* Apply class_weight='balanced'
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(class_weight='balanced')

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(class_weight='balanced') 
* Resampling
    from imblearn.over_sampling import SMOTE

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
* Consider threshold tuning
    threshold = 0.3


=== Potential Leakage Feature ====
length_of_stay_days:
apache_score
sofa_score
severity_combined
