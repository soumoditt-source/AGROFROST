import os
import json
import glob

def get_latest_reports():
    files = sorted(glob.glob('Drone image/*/analysis_report_*.json'), key=os.path.getmtime)
    for f in files:
        with open(f, 'r') as j:
            data = json.load(j)
            print(f"FOLDER: {data['folder']}")
            print(f"Processed at: {data['processed_at']}")
            print(f"Total Pits: {data['metrics']['total']}")
            print(f"Survival Rate: {data['metrics']['rate']:.2f}%")
            print("-" * 20)

if __name__ == "__main__":
    get_latest_reports()
