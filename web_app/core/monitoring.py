import time
import threading
import psutil
import subprocess
import os
import csv
from datetime import datetime
from typing import Optional, Tuple

class PerformanceMonitor:
    def __init__(self, output_dir: str, run_id: str) -> None:
        self.log_dir = output_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"perf_{run_id}.csv")
        self.stop_event = threading.Event()
        self.thread = None

    def start(self, interval: float = 1.0) -> None:
        self.stop_event.clear()
        # Initialize CSV header
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'CPU_Usage_Percent', 'RAM_Usage_Percent', 'GPU_Usage_Percent', 'VRAM_Used_MB', 'VRAM_Total_MB'])
        
        self.thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.thread.start()
        print(f"[MONITOR] Performance logging started: {self.log_path}")

    def stop(self) -> None:
        if self.thread:
            self.stop_event.set()
            self.thread.join()
            print("[MONITOR] Performance logging stopped.")

    def _get_gpu_stats(self) -> Tuple[float, float, float]:
        try:
            # Query nvidia-smi for utilization and memory
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                # Output format: "utilization, memory.used, memory.total"
                # Example: "0, 485, 8192"
                line = result.stdout.strip()
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 3:
                    return float(parts[0]), float(parts[1]), float(parts[2])
        except Exception:
            pass
        return 0.0, 0.0, 0.0

    def _monitor_loop(self, interval: float) -> None:
        while not self.stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                gpu_util, vram_used, vram_total = self._get_gpu_stats()
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(self.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, cpu, ram, gpu_util, vram_used, vram_total])
                
                time.sleep(interval)
            except Exception as e:
                print(f"[MONITOR] Error: {e}")
                time.sleep(interval)
