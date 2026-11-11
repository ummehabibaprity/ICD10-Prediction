import os
os.environ["RUN_PHASE"] = "none"
from revised_pipeline import train_and_evaluate, write_tables

train_and_evaluate(
    "emilyalsentzer/Bio_ClinicalBERT",
    "clinicalbert",
    use_focal=False,
    use_weighted=False,
    tag="baseline_clinicalbert",
)
write_tables()
