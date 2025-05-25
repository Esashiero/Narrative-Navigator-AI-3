import threading
import json
import queue
import time
import ollama
from PyQt5.QtCore import QThread, pyqtSignal

class ChatThread(QThread):
    chat_response = pyqtSignal(str)
    chat_log = pyqtSignal(dict) 

    def __init__(self, transcript_getter, entities_getter, content_title, external_context):
        super().__init__()
        self.chat_queue = queue.Queue()
        self.running = False
        self.transcript_getter = transcript_getter
        self.entities_getter = entities_getter
        self.content_title = content_title
        self.external_context = external_context

        # ADDED: Event for responsive shutdown
        self._stop_event = threading.Event()

        self.system_prompt = """
You are an AI assistant helping a user understand a story by answering questions based on the transcript history and a narrative cheat sheet.

The content is titled: "{content_title}".
External context about the content:
---
{external_context}
---

Answer user questions using the provided transcript history and cheat sheet. Provide detailled, relevant answers in plain text.
""".format(content_title=self.content_title, external_context=self.external_context)

    def add_chat_query(self, query):
        self.chat_queue.put(query)

    def run(self):
        self.running = True
        # ADDED: Clear stop event at the start of run
        self._stop_event.clear()

        while self.running:
            try:
                # Changed to get with timeout to be responsive to stop signals
                query = self.chat_queue.get(timeout=0.1) 
                
                transcripts = self.transcript_getter()
                recent_transcripts = transcripts[-5:] if len(transcripts) > 5 else transcripts
                current_entities = self.entities_getter()
                current_entities_json = json.dumps(current_entities, indent=2)

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content":
                     f"Transcript history: {recent_transcripts}\n"
                     f"Current narrative cheat sheet: {current_entities_json}\n"
                     f"User question: {query}"}
                ]
                self.chat_log.emit({"type": "chat_prompt", "message": "Chat prompt sent:", "data": messages})
                try:
                    response = ollama.chat(model="llama3.2:latest", messages=messages, stream=False)
                    content = response['message']['content']
                    self.chat_response.emit(content)
                    self.chat_log.emit({"type": "chat_response", "message": "Chat response received.", "data": content})
                except Exception as e:
                    self.chat_response.emit(f"Error: Unable to process query - {str(e)}")
                    self.chat_log.emit({"type": "error", "message": f"Chat Error: {str(e)}"})
            except queue.Empty:
                # If queue is empty (no chat query), check if stop event is set
                if self._stop_event.is_set():
                    self.chat_log.emit({"type": "status", "message": "Chat Thread received stop signal during idle, exiting."})
                    break # Exit the loop immediately
            # REMOVED: original time.sleep(0.5) as get(timeout) handles idle waiting.
            # If the queue wasn't empty, processing might take time, so no additional sleep is needed.

    def stop(self):
        self.running = False
        # ADDED: Set stop event to signal termination
        self._stop_event.set()