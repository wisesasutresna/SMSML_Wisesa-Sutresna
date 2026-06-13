import time
import argparse
import pandas as pd
import requests

def dapatkan_data_fitur():
    try:
        df = pd.read_csv("diabetes_preprocessing.csv")
    except FileNotFoundError:
        df = pd.read_csv("diabetes_preprocessing")
    return df.drop(columns=["Outcome"])

def kirim_permintaan_inference(total_request, jeda_waktu, pemicu_error):
    X = dapatkan_data_fitur()
    endpoint_url = "http://127.0.0.1:8003/predict"
    
    print(f"Memulai kirim data diabetes ({total_request} request) ke {endpoint_url}...")
    for counter in range(total_request):
        sample = X.sample(n=1)
        
        payload_data = {
            "dataframe_split": {
                "columns": list(sample.columns),
                "data": sample.values.tolist()
            }
        }
        
        # Kirim data kosong atau salah struktur untuk trigger fail counter
        if pemicu_error and counter % 5 == 0:
            payload_data = {"data_salah": None}
            print(f"[{counter+1}/{total_request}] Mengirim payload error terencana...")
        else:
            print(f"[{counter+1}/{total_request}] Mengirim payload inference sukses...")
            
        try:
            res = requests.post(endpoint_url, json=payload_data)
            print(f"Hasil: HTTP {res.status_code} - {res.json()}")
        except Exception as error:
            print(f"Error saat menghubungi port exporter: {error}")
            
        time.sleep(jeda_waktu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_requests", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--trigger_errors", action="store_true")
    args = parser.parse_args()
    
    kirim_permintaan_inference(args.num_requests, args.delay, args.trigger_errors)
