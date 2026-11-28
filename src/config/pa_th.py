from pathlib import Path



pro_path=Path(__file__).resolve().parents[2]

data_path=pro_path/"DATA"/"RAW"
artifacts_path=pro_path/"artifacts"
charts_path=artifacts_path/"charts"
report_path=artifacts_path/"reports"
model_path=artifacts_path/"models"
model_tun_path=model_path/"model_tuning"
model_train_path=model_path/"model_training"
