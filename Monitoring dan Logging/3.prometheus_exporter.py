import time
import psutil
import requests
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Definisi metrik API
METRIC_REQ_TOTAL = Counter('wisesa_sutresna_http_requests_total', 'Total Request HTTP Masuk')
METRIC_REQ_LATENCY = Histogram('wisesa_sutresna_http_request_duration_seconds', 'Durasi Latensi Request')
METRIC_THROUGHPUT = Counter('wisesa_sutresna_http_requests_throughput', 'Throughput per detik')
METRIC_REQ_FAILED = Counter('wisesa_sutresna_http_requests_failed_total', 'Total Request HTTP Gagal')

# Definisi metrik hasil prediksi model
METRIC_PRED_CLASS_0 = Counter('wisesa_sutresna_predictions_class_0_total', 'Prediksi Kelas 0 (Non-Diabetes)')
METRIC_PRED_CLASS_1 = Counter('wisesa_sutresna_predictions_class_1_total', 'Prediksi Kelas 1 (Diabetes)')

# Definisi metrik status infrastruktur OS
METRIC_CPU_PCT = Gauge('wisesa_sutresna_system_cpu_usage', 'Persentase Utilisasi CPU')
METRIC_RAM_PCT = Gauge('wisesa_sutresna_system_ram_usage', 'Persentase Utilisasi RAM')
METRIC_DISK_PCT = Gauge('wisesa_sutresna_system_disk_usage', 'Persentase Utilisasi Disk')
METRIC_NET_SENT = Counter('wisesa_sutresna_system_network_sent_bytes', 'Total Byte Jaringan Dikirim')
METRIC_NET_RECV = Counter('wisesa_sutresna_system_network_recv_bytes', 'Total Byte Jaringan Diterima')
METRIC_PROC_THREADS = Gauge('wisesa_sutresna_system_process_threads', 'Jumlah Thread Aktif pada Proses')

def perbarui_metrik_sistem():
    """Mengambil data performa server OS terkini menggunakan psutil."""
    METRIC_CPU_PCT.set(psutil.cpu_percent())
    METRIC_RAM_PCT.set(psutil.virtual_memory().percent)
    METRIC_DISK_PCT.set(psutil.disk_usage('/').percent)
    
    net_io = psutil.net_io_counters()
    METRIC_NET_SENT.inc(max(0, net_io.bytes_sent - METRIC_NET_SENT._value.get()))
    METRIC_NET_RECV.inc(max(0, net_io.bytes_recv - METRIC_NET_RECV._value.get()))
    
    proc = psutil.Process()
    METRIC_PROC_THREADS.set(proc.num_threads())

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Perbarui metrik sebelum diekspor
    perbarui_metrik_sistem()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def handle_predict():
    start_time = time.time()
    METRIC_REQ_TOTAL.inc()
    METRIC_THROUGHPUT.inc()
    
    data = request.get_json()
    model_service_url = "http://127.0.0.1:5003/invocations"
    
    try:
        res = requests.post(model_service_url, json=data)
        if res.status_code != 200:
            METRIC_REQ_FAILED.inc()
            return jsonify({"error": f"Error response dari model service: {res.status_code}"}), res.status_code
            
        latency = time.time() - start_time
        METRIC_REQ_LATENCY.observe(latency)
        
        # Iterasi hasil klasifikasi dari model mlflow
        predictions = res.json().get("predictions", [])
        for pred in predictions:
            if pred == 0:
                METRIC_PRED_CLASS_0.inc()
            elif pred == 1:
                METRIC_PRED_CLASS_1.inc()
                
        return jsonify(res.json())
        
    except Exception as err:
        METRIC_REQ_FAILED.inc()
        return jsonify({"error": str(err)}), 500

if __name__ == '__main__':
    # Running di port 8003
    app.run(host='127.0.0.1', port=8003)
