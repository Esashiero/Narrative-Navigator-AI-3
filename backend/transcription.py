import numpy as np
import whisper
import threading
import time
import queue
from PyQt5.QtCore import QThread, pyqtSignal
from constants import TRANSCRIPT_CHUNK_DURATION_SECONDS

class TranscriptionThread(QThread):
    transcription = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        try:
            self.model = whisper.load_model("small.en")
        except Exception as e:
            self.error_signal.emit(f"Failed to load Whisper model: {e}")
            self.model = None

        self.audio_buffer = np.array([], dtype=np.float32)
        self.running = False
        self.buffer_lock = threading.Lock()
        self._stop_event = threading.Event()

    def run(self):
        self.running = True
        self._stop_event.clear()

        while self.running:
            audio_to_process = None
            with self.buffer_lock:
                if len(self.audio_buffer) >= 16000 * TRANSCRIPT_CHUNK_DURATION_SECONDS: 
                    audio_to_process = self.audio_buffer[:16000 * TRANSCRIPT_CHUNK_DURATION_SECONDS].copy()
                    self.audio_buffer = self.audio_buffer[16000 * TRANSCRIPT_CHUNK_DURATION_SECONDS:]
            
            if audio_to_process is not None:
                try:
                    if audio_to_process.dtype != np.float32:
                        audio_to_process = audio_to_process.astype(np.float32) / 32768.0

                    result = self.model.transcribe(audio_to_process, language="en")
                    if result["text"].strip():
                        self.transcription.emit(result["text"])
                except Exception as e:
                    self.error_signal.emit(f"Transcription Error: {e}")
                    # print(f"Transcription Error: {e}", file=sys.stderr) # Removed sys.stderr import
            time.sleep(0.1)
            if self._stop_event.is_set():
                break

    def add_audio(self, audio_data):
        with self.buffer_lock:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            self.audio_buffer = np.append(self.audio_buffer, audio_data.flatten())

    def stop(self):
        self.running = False
        self._stop_event.set()