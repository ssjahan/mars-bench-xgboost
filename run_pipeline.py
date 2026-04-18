import subprocess
import sys
import os

def run_step(script_path, description):
    print("\n" + "=" * 60)
    print(f"Executing: {description}")
    print("=" * 60)
    
    # Run the script using the current Python executable (e.g. from venv)
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {description}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("Starting Mars-Bench XGBoost End-to-End Pipeline...")
    
    run_step("data/download_data.py", "Step 1: Download and Prepare Data")
    run_step("src/extract_features.py", "Step 2: Extract ViT Features")
    run_step("src/train_xgboost.py", "Step 3: Train XGBoost Models")
    run_step("src/evaluate.py", "Step 4: Rigorous 7-Seed Evaluation")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
