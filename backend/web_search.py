import json # Not directly used but often helpful for debugging DDGS results
from duckduckgo_search import DDGS
from PyQt5.QtCore import QThread, pyqtSignal

class WebSearchThread(QThread):
    context_ready = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, title):
        super().__init__()
        self.title = title

    def run(self):
        try:
            search_query = f"{self.title} plot summary OR overview"
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=5))

            context_lines = []
            for r in results:
                if r.get('body'):
                    context_lines.append(f"- {r['title']}: {r['body']}")
                elif r.get('link'):
                    context_lines.append(f"- {r['title']} ({r['link']})")
            
            full_context = "\n".join(context_lines)

            if not full_context:
                self.error_signal.emit(f"Warning: No significant web context found for '{self.title}'. LLM may operate with limited external knowledge.")
                full_context = f"No specific plot context found for the content titled '{self.title}'."

            self.context_ready.emit(full_context)
        except Exception as e:
            self.error_signal.emit(f"Error during web search for '{self.title}': {str(e)}. LLM will operate without external context.")
            self.context_ready.emit(f"Web search failed. No external context provided for '{self.title}'.")