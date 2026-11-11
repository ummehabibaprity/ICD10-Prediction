import os
os.environ["RUN_PHASE"] = "none"
from revised_pipeline import train_and_evaluate, write_tables

name = "distilbert-base-uncased"
train_and_evaluate(name, "distilbert", False, False, tag="ablation_baseline")
train_and_evaluate(name, "distilbert", True,  False, tag="ablation_focal")
train_and_evaluate(name, "distilbert", False, True,  tag="ablation_weighted")
train_and_evaluate(name, "distilbert", True,  True,  tag="ablation_focal_weighted")
write_tables()
