import sys
import numpy as np
import sounddevice as sd
import queue
import threading
from PyQt5.QtCore import QThread, pyqtSignal
from constants import TRANSCRIPT_CHUNK_DURATION_SECONDS # Not directly used here, but good practice to keep context

class AudioCaptureThread(QThread):
    audio_data = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, device_id=3, samplerate=16000):
        super().__init__()
        self.device_id = device_id
        self.samplerate = samplerate
        self._running = False
        self._stop_event = threading.Event()
        self.audio_queue = queue.Queue()

    def run(self):
        self._running = True
        self._stop_event.clear()

        def callback(indata, frames, time_info, status):
            if status:
                pass
            if self._running:
                self.audio_queue.put(indata.copy())

        try:
            devices = sd.query_devices()
            if self.device_id >= len(devices):
                raise ValueError(f"Device ID {self.device_id} is out of range. Available devices: {len(devices)}")
            device_info = sd.query_devices(self.device_id)
            if device_info['max_input_channels'] < 1:
                raise ValueError(f"Device {self.device_id} does not support audio input")
            
            with sd.InputStream(samplerate=self.samplerate,
                              device=self.device_id,
                              channels=1,
                              callback=callback):
                while self._running:
                    try:
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        self.audio_data.emit(audio_chunk)
                    except queue.Empty:
                        if self._stop_event.is_set():
                            break
        except Exception as e:
            error_msg = f"Audio Capture Critical Error: {str(e)}"
            if "device" in str(e).lower():
                error_msg += f"\nAvailable audio devices:\n"
                for i, dev in enumerate(sd.query_devices()):
                    if dev['max_input_channels'] > 0:
                        error_msg += f"ID {i}: {dev['name']}\n"
            self.error_signal.emit(error_msg)
            print(error_msg, file=sys.stderr)
        finally:
            self._running = False
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

    def stop(self):
        self._running = False
        self._stop_event.set()