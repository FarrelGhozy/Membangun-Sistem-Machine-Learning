import time
import random
import psutil
import platform
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import os


CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Memory usage percentage')
MEMORY_AVAILABLE = Gauge('system_memory_available_bytes', 'Available memory in bytes')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk usage percentage')
NETWORK_BYTES_SENT = Gauge('system_network_bytes_sent', 'Network bytes sent')
NETWORK_BYTES_RECV = Gauge('system_network_bytes_recv', 'Network bytes received')
PROCESS_COUNT = Gauge('system_process_count', 'Number of running processes')
PYTHON_THREAD_COUNT = Gauge('system_python_threads', 'Number of Python threads')
UPTIME_SECONDS = Gauge('system_uptime_seconds', 'System uptime in seconds')
LOAD_AVERAGE = Gauge('system_load_average', 'System load average (1 min)')

REQUEST_DURATION = Histogram(
    'exporter_request_duration_seconds',
    'Exporter scrape duration',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

SCRAPE_COUNT = Counter('exporter_scrapes_total', 'Total number of exporter scrapes')

START_TIME = time.time()


def collect_system_metrics():
    SCRAPE_COUNT.inc()
    start = time.time()

    try:
        CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
        memory = psutil.virtual_memory()
        MEMORY_USAGE.set(memory.percent)
        MEMORY_AVAILABLE.set(memory.available)
        DISK_USAGE.set(psutil.disk_usage('/').percent)

        net = psutil.net_io_counters()
        NETWORK_BYTES_SENT.set(net.bytes_sent)
        NETWORK_BYTES_RECV.set(net.bytes_recv)

        PROCESS_COUNT.set(len(psutil.pids()))
        UPTIME_SECONDS.set(time.time() - START_TIME)
        LOAD_AVERAGE.set(psutil.getloadavg()[0])

        import threading
        PYTHON_THREAD_COUNT.set(threading.active_count())

    except Exception as e:
        print(f"[ERROR] Failed to collect metrics: {e}")

    duration = time.time() - start
    REQUEST_DURATION.observe(duration)


if __name__ == '__main__':
    PORT = 8001
    start_http_server(PORT)
    print(f"[INFO] Prometheus exporter started on port {PORT}")
    print(f"[INFO] Dashboard: fazyfif")

    while True:
        collect_system_metrics()
        time.sleep(10)
