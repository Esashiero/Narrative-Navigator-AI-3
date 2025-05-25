import sys
import json
import numpy as np
import time 
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextEdit, QTableWidget,
    QTableWidgetItem, QInputDialog, QSplitter, QLineEdit,
    QLabel, QFrame, QTabWidget, QScrollArea, QSizePolicy,
    QApplication, QStyle
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QIcon

from backend.audio_capture import AudioCaptureThread
from backend.transcription import TranscriptionThread
from backend.web_search import WebSearchThread
from backend.llm_processing import LLMThread
from backend.chat_agent import ChatThread
from constants import TRANSCRIPT_CHUNK_DURATION_SECONDS

class NarrativeNavigator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Narrative Navigator AI")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)

        self.audio_thread = AudioCaptureThread()
        self.transcription_thread = TranscriptionThread()
        self.llm_thread = LLMThread()
        self.chat_thread = None
        self.web_search_thread = None
        self.content_title = ""
        self.minimum_display_score = 3 
        
        self.cheat_sheet_column_widths = {} 

        self.init_ui()

        self.audio_thread.audio_data.connect(self.transcription_thread.add_audio)
        self.audio_thread.error_signal.connect(lambda msg: self.update_llm_log_tabs({"type": "error", "message": msg})) 
        self.transcription_thread.transcription.connect(self.handle_transcription)
        self.transcription_thread.error_signal.connect(lambda msg: self.update_llm_log_tabs({"type": "error", "message": msg}))
        self.llm_thread.entities_updated.connect(self.update_entity_displays) 
        self.llm_thread.llm_log.connect(self.update_llm_log_tabs)

        self.get_content_title_and_context()

    def init_ui(self):
        main_container = QWidget()
        self.setCentralWidget(main_container)
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header_widget = QWidget()
        header_widget.setObjectName("headerWidget")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 10, 20, 10)
        header_layout.setSpacing(15)

        app_logo = QLabel()
        app_logo.setPixmap(self.style().standardIcon(QStyle.SP_ComputerIcon).pixmap(QSize(32, 32)))
        
        app_name_label = QLabel("Narrative Navigator")
        app_name_label.setObjectName("appNameLabel")

        self.analysis_status_label = QLabel("Analyzing: [FULL] Beyond Boiling Point - Gordon Ramsay documentary (2000)")
        self.analysis_status_label.setObjectName("analysisStatusLabel")
        self.analysis_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.analysis_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        header_layout.addWidget(app_logo)
        header_layout.addWidget(app_name_label)
        header_layout.addWidget(self.analysis_status_label)

        self.volume_button = QPushButton()
        self.volume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        self.volume_button.setObjectName("iconButton")
        self.volume_button.setIconSize(QSize(20, 20))
        self.volume_button.setFixedSize(36, 36)

        self.toggle_button = QPushButton("Stop Recording")
        self.toggle_button.setObjectName("stopRecordingButton")
        self.toggle_button.clicked.connect(self.toggle_processing)
        self.toggle_button.setEnabled(False)

        self.close_button = QPushButton()
        self.close_button.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        self.close_button.setObjectName("iconButton")
        self.close_button.setIconSize(QSize(20,20))
        self.close_button.setFixedSize(36,36)
        self.close_button.clicked.connect(self.close)

        header_layout.addWidget(self.volume_button)
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.close_button)

        main_layout.addWidget(header_widget)

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setContentsMargins(10, 0, 10, 10) 

        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("mainTabWidget")
        content_splitter.addWidget(self.tab_widget)

        self.overview_tab_page = self._create_overview_tab()
        self.story_elements_tab_page = self._create_story_elements_tab()
        self.live_transcript_tab_page = self._create_live_transcript_tab()
        self.ai_chat_tab_page = self._create_ai_chat_tab()
        self.llm_log_tab_page = self._create_llm_log_tab() 

        self.tab_widget.addTab(self.overview_tab_page, "Overview")
        self.tab_widget.addTab(self.story_elements_tab_page, "Story Elements")
        self.tab_widget.addTab(self.live_transcript_tab_page, "Live Transcript")
        self.tab_widget.addTab(self.ai_chat_tab_page, "AI Chat")
        self.tab_widget.addTab(self.llm_log_tab_page, "LLM Log")

        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(10)

        cheat_sheet_label = QLabel("Narrative Cheat Sheet")
        cheat_sheet_label.setObjectName("sectionTitle") 
        right_layout.addWidget(cheat_sheet_label)

        self.cheat_sheet_table = QTableWidget() 
        self.cheat_sheet_table.setColumnCount(6)
        self.cheat_sheet_table.setHorizontalHeaderLabels(["Name", "Type", "Description", "Base Score", "Mentions", "Current Score"])
        self.cheat_sheet_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cheat_sheet_table.verticalHeader().setVisible(False)
        self.cheat_sheet_table.setAlternatingRowColors(True)
        self.cheat_sheet_table.setObjectName("cheatSheetTable") 
        right_layout.addWidget(self.cheat_sheet_table)

        self.cheat_sheet_column_widths = {
            0: 120,   # Name
            1: 100,   # Type
            2: 250,   # Description
            3: 80,    # Base Score
            4: 70,    # Mentions
            5: 100    # Current Score
        }
        for i, width in self.cheat_sheet_column_widths.items():
            self.cheat_sheet_table.setColumnWidth(i, width)

        self.cheat_sheet_table.horizontalHeader().sectionResized.connect(self._on_cheat_sheet_column_resized)

        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([800, 400]) 
        main_layout.addWidget(content_splitter)

    def _on_cheat_sheet_column_resized(self, logicalIndex, oldSize, newSize):
        """Slot to remember user-resized column widths."""
        self.cheat_sheet_column_widths[logicalIndex] = newSize

    def _create_overview_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        title_label = QLabel("Overview")
        title_label.setObjectName("sectionTitle")
        tab_layout.addWidget(title_label)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)
        
        self.characters_count_label = QLabel("0") 
        self.locations_count_label = QLabel("0") 
        self.transcript_lines_count_label = QLabel("0")
        self.total_elements_count_label = QLabel("0") 

        metrics_layout.addWidget(self._create_summary_card(self.characters_count_label, "Characters", QStyle.SP_MessageBoxInformation)) 
        metrics_layout.addWidget(self._create_summary_card(self.locations_count_label, "Locations", QStyle.SP_DirIcon)) 
        metrics_layout.addWidget(self._create_summary_card(self.transcript_lines_count_label, "Transcript Lines", QStyle.SP_FileIcon)) 
        metrics_layout.addWidget(self._create_summary_card(self.total_elements_count_label, "Total Elements", QStyle.SP_MessageBoxQuestion)) 
        tab_layout.addLayout(metrics_layout)

        recent_activity_widget = QFrame()
        recent_activity_widget.setProperty("class", "contentCard")
        recent_activity_layout = QVBoxLayout(recent_activity_widget)
        recent_activity_layout.setContentsMargins(15,15,15,15)
        recent_activity_layout.setSpacing(10)

        recent_activity_layout.addWidget(QLabel("Recent Activity"))
        self.recent_activity_display = QTextEdit()
        self.recent_activity_display.setReadOnly(True)
        self.recent_activity_display.setPlaceholderText("Latest story elements and transcript updates...")
        self.recent_activity_display.setFixedHeight(150)
        recent_activity_layout.addWidget(self.recent_activity_display)
        tab_layout.addWidget(recent_activity_widget)

        key_characters_widget = QFrame()
        key_characters_widget.setProperty("class", "contentCard")
        key_characters_layout = QVBoxLayout(key_characters_widget)
        key_characters_layout.setContentsMargins(15,15,15,15)
        key_characters_layout.setSpacing(10)
        
        key_characters_layout.addWidget(QLabel("Key Characters"))
        self.key_characters_container_layout = QVBoxLayout()
        self.key_characters_container_layout.setContentsMargins(0,0,0,0)
        self.key_characters_container_layout.setSpacing(8)
        
        self.key_characters_container_layout.addStretch() 

        key_characters_layout.addLayout(self.key_characters_container_layout)
        tab_layout.addWidget(key_characters_widget)

        tab_layout.addStretch() 
        return tab_page

    def _create_summary_card(self, count_label_ref, label_text, icon_style_hint):
        card = QFrame()
        card.setProperty("class", "contentCard")
        card.setFixedSize(160, 120)
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignCenter)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(5)

        icon_label = QLabel()
        icon_label.setPixmap(self.style().standardIcon(icon_style_hint).pixmap(QSize(30, 30)))
        card_layout.addWidget(icon_label, alignment=Qt.AlignCenter)

        count_label_ref.setStyleSheet("font-size: 28px; font-weight: bold; color: #6a0dad;")
        card_layout.addWidget(count_label_ref, alignment=Qt.AlignCenter)

        text_label = QLabel(label_text)
        text_label.setStyleSheet("font-size: 13px; color: #555555; font-weight: 500;")
        card_layout.addWidget(text_label, alignment=Qt.AlignCenter)
        return card

    def _create_story_elements_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        title_label = QLabel("Story Elements")
        title_label.setObjectName("sectionTitle")
        tab_layout.addWidget(title_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)

        self.story_elements_content_widget = QWidget()
        self.story_elements_container_layout = QVBoxLayout(self.story_elements_content_widget)
        self.story_elements_container_layout.setContentsMargins(0, 0, 0, 0)
        self.story_elements_container_layout.setSpacing(10)

        self.story_elements_container_layout.addStretch() 

        scroll_area.setWidget(self.story_elements_content_widget)
        tab_layout.addWidget(scroll_area)
        
        return tab_page
    
    def _create_entity_card(self, entity_data, first_mentioned_time="00:00:00"):
        card = QFrame()
        card.setProperty("class", "storyElementCard")

        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0) 
        card_layout.setSpacing(5) 

        icon_label = QLabel()
        if entity_data["type"] == "Characters":
            icon_label.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxInformation).pixmap(QSize(16, 16)))
        elif entity_data["type"] == "Locations":
            icon_label.setPixmap(self.style().standardIcon(QStyle.SP_DirIcon).pixmap(QSize(16, 16)))
        elif entity_data["type"] == "Organizations":
            icon_label.setPixmap(self.style().standardIcon(QStyle.SP_DesktopIcon).pixmap(QSize(16, 16)))
        else: 
            icon_label.setPixmap(self.style().standardIcon(QStyle.SP_FileIcon).pixmap(QSize(16, 16)))
        card_layout.addWidget(icon_label)

        element_name = QLabel(entity_data["name"])
        element_name.setProperty("property", "nameLabel") 
        card_layout.addWidget(element_name)
        
        category_tag = QLabel(entity_data["type"])
        category_tag.setProperty("class", "categoryTag") 
        category_tag.setObjectName(f"categoryTag-{entity_data['type'].replace(' ', '')}")
        card_layout.addWidget(category_tag)

        separator = QLabel("—") 
        separator.setStyleSheet("color: #aaaaaa; margin: 0 5px;")
        card_layout.addWidget(separator)

        description_text = entity_data.get("description", "No description available")
        max_desc_length = 80 
        if len(description_text) > max_desc_length:
            description_text = description_text[:max_desc_length].strip() + "..."
        
        description = QLabel(description_text)
        description.setProperty("property", "descriptionLabel") 
        description.setWordWrap(False) 
        description.setTextFormat(Qt.PlainText) 
        description.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) 
        card_layout.addWidget(description)

        card_layout.addStretch() 

        first_mentioned = QLabel(f"First mentioned: {first_mentioned_time}")
        first_mentioned.setProperty("property", "timeLabel") 
        card_layout.addWidget(first_mentioned)
        
        return card

    def _create_live_transcript_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        header_layout = QHBoxLayout()
        mic_icon = QLabel()
        mic_icon.setPixmap(self.style().standardIcon(QStyle.SP_MediaVolume).pixmap(QSize(24, 24)))
        header_layout.addWidget(mic_icon)
        
        title_label = QLabel("Live Transcript")
        title_label.setObjectName("sectionTitle")
        header_layout.addWidget(title_label)

        self.processing_tag = QLabel("Processing")
        self.processing_tag.setProperty("class", "processingTag")
        self.processing_tag.setVisible(False)
        header_layout.addWidget(self.processing_tag)
        header_layout.addStretch()
        tab_layout.addLayout(header_layout)

        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setPlaceholderText("Live transcription will appear here...")
        tab_layout.addWidget(self.transcript_display)
        
        return tab_page

    def _create_ai_chat_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        header_layout = QHBoxLayout()
        ai_icon = QLabel()
        ai_icon.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxQuestion).pixmap(QSize(24, 24)))
        header_layout.addWidget(ai_icon)
        
        title_label = QLabel("AI Story Assistant")
        title_label.setObjectName("sectionTitle")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        tab_layout.addLayout(header_layout)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Chat with the AI about the story...")
        tab_layout.addWidget(self.chat_display)

        chat_input_container = QFrame()
        chat_input_container.setObjectName("chatInputContainer")
        chat_input_layout = QHBoxLayout(chat_input_container)
        chat_input_layout.setContentsMargins(0, 0, 0, 0)
        chat_input_layout.setSpacing(0)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask about characters, plot, or story elements...")
        self.chat_input.returnPressed.connect(self.send_chat_query)
        chat_input_layout.addWidget(self.chat_input)

        send_button = QPushButton()
        send_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        send_button.setIconSize(QSize(20, 20))
        send_button.setFixedSize(40, 40)
        send_button.setObjectName("sendButton")
        send_button.clicked.connect(self.send_chat_query)
        chat_input_layout.addWidget(send_button)

        tab_layout.addWidget(chat_input_container)
        return tab_page

    def _create_llm_log_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        title_label = QLabel("LLM Processing Log")
        title_label.setObjectName("sectionTitle")
        tab_layout.addWidget(title_label)

        self.llm_log_tabs = QTabWidget()
        self.llm_log_tabs.setObjectName("llmLogSubTabs")

        raw_log_tab = QWidget()
        raw_log_layout = QVBoxLayout(raw_log_tab)
        self.llm_raw_log_display = QTextEdit()
        self.llm_raw_log_display.setReadOnly(True)
        self.llm_raw_log_display.setPlaceholderText("Raw LLM prompts and responses will appear here...")
        raw_log_layout.addWidget(self.llm_raw_log_display)
        self.llm_log_tabs.addTab(raw_log_tab, "Raw Interactions")

        parsed_entities_tab = QWidget()
        parsed_entities_layout = QVBoxLayout(parsed_entities_tab)
        self.llm_parsed_entities_table = QTableWidget()
        self.llm_parsed_entities_table.setColumnCount(4) 
        self.llm_parsed_entities_table.setHorizontalHeaderLabels(["Name", "Type", "Description", "Base Score"])
        self.llm_parsed_entities_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.llm_parsed_entities_table.verticalHeader().setVisible(False)
        self.llm_parsed_entities_table.setAlternatingRowColors(True)
        self.llm_parsed_entities_table.horizontalHeader().setStretchLastSection(True)
        self.llm_parsed_entities_table.setObjectName("llmParsedEntitiesTable")
        parsed_entities_layout.addWidget(self.llm_parsed_entities_table)
        self.llm_log_tabs.addTab(parsed_entities_tab, "Parsed Entities (Latest)")

        errors_warnings_tab = QWidget()
        errors_warnings_layout = QVBoxLayout(errors_warnings_tab)
        self.llm_error_warnings_display = QTextEdit()
        self.llm_error_warnings_display.setReadOnly(True)
        self.llm_error_warnings_display.setPlaceholderText("LLM-related errors and warnings will appear here...")
        self.llm_error_warnings_display.setStyleSheet("QTextEdit { color: #8B0000; }") 
        errors_warnings_layout.addWidget(self.llm_error_warnings_display)
        self.llm_log_tabs.addTab(errors_warnings_tab, "Errors & Warnings")

        tab_layout.addWidget(self.llm_log_tabs)
        return tab_page

    def get_content_title_and_context(self):
        # Create a QInputDialog instance to set window flags
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Content Title")
        dialog.setLabelText("Please enter the title of the content you are watching:")
        dialog.setTextEchoMode(QLineEdit.Normal)
        dialog.setTextValue("The Outlast Trials Lore") # Default text
        
        # Set the window flag to always stay on top
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        
        # Execute the dialog and get the result
        ok = dialog.exec_() # Use exec_() for modal dialogs
        title = dialog.textValue() # Get the entered text
        
        if ok and title:
            self.content_title = title.strip()
            self.analysis_status_label.setText(f"Analyzing: [FULL] {self.content_title}")
            if not self.content_title:
                self.update_llm_log_tabs({"type": "error", "message": "Empty content title provided. LLM will operate without specific external context."})
                self.llm_thread.set_external_context("No external context provided by user.")
                self.init_chat_thread()
                self.toggle_button.setEnabled(True)
                return

            self.llm_thread.set_content_title(self.content_title)
            self.update_llm_log_tabs({"type": "status", "message": f"Searching for external context for: '{self.content_title}'..."})
            self.web_search_thread = WebSearchThread(self.content_title)
            self.web_search_thread.context_ready.connect(self.set_llm_external_context)
            self.web_search_thread.error_signal.connect(lambda msg: self.update_llm_log_tabs({"type": "error", "message": msg})) 
            self.web_search_thread.start()
        else:
            self.update_llm_log_tabs({"type": "status", "message": "No content title provided. LLM will operate without specific external context."})
            self.llm_thread.set_external_context("No external context provided by user.")
            self.init_chat_thread()
            self.toggle_button.setEnabled(True)

    def set_llm_external_context(self, context):
        self.llm_thread.set_external_context(context)
        self.update_llm_log_tabs({"type": "status", "message": f"External context loaded for LLM (showing first 500 chars):\n<pre>{context[:500]}...</pre>"})
        if len(context) > 500:
            self.update_llm_log_tabs({"type": "status", "message": f"...\n(Full context is {len(context)} characters long)"})
        self.init_chat_thread()
        self.toggle_button.setEnabled(True)
        self.toggle_button.setText("Start Recording")
        self.update_llm_log_tabs({"type": "status", "message": "You can now click 'Start Recording' to begin audio processing and entity extraction."})

    def init_chat_thread(self):
        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.stop()
            self.chat_thread.wait()

        self.chat_thread = ChatThread(
            transcript_getter=self.llm_thread.get_transcriptions,
            entities_getter=self.llm_thread.get_entities,
            content_title=self.content_title,
            external_context=self.llm_thread.external_context
        )
        self.chat_thread.chat_response.connect(self.display_chat_response)
        self.chat_thread.chat_log.connect(self.update_llm_log_tabs) 
        self.chat_thread.start()
        self.update_llm_log_tabs({"type": "status", "message": "Chat thread initialized."})

    def toggle_processing(self):
        if self.toggle_button.text() == "Start Recording":
            self.audio_thread.start()
            self.transcription_thread.start()
            self.llm_thread.start()
            self.toggle_button.setText("Stop Recording")
            self.processing_tag.setVisible(True)
            self.update_llm_log_tabs({"type": "status", "message": "Processing started."})
        else:
            self.stop_processing()
            self.toggle_button.setText("Start Recording")
            self.processing_tag.setVisible(False)
            self.update_llm_log_tabs({"type": "status", "message": "Processing stopped."})

    def stop_processing(self):
        self.llm_thread.stop()
        self.transcription_thread.stop()
        self.audio_thread.stop()
        self.llm_thread.wait()
        self.transcription_thread.wait()
        self.audio_thread.wait()
        self.update_llm_log_tabs({"type": "status", "message": "All processing threads stopped."})

    def handle_transcription(self, text):
        if self.transcript_display:
            self.transcript_display.append(text)
            self.transcript_display.verticalScrollBar().setValue(self.transcript_display.verticalScrollBar().maximum())
            current_lines = int(self.transcript_lines_count_label.text())
            self.transcript_lines_count_label.setText(str(current_lines + 1))

        self.llm_thread.add_transcription(text) 
        self.recent_activity_display.append(f"Transcript: {text[:80].strip()}...")
        self.recent_activity_display.verticalScrollBar().setValue(self.recent_activity_display.verticalScrollBar().maximum())

    def update_entity_displays(self, all_entities):
        for e in all_entities:
            e["current_importance_score"] = e.get("base_importance_score", 0) + e.get("mention_count", 0)

        scores = [e["current_importance_score"] for e in all_entities]
        
        display_threshold = self.minimum_display_score 
        
        if scores:
            median_score = np.median(scores)
            display_threshold = max(self.minimum_display_score, median_score)
            self.update_llm_log_tabs({"type": "debug", "message": f"Calculated median importance score: {median_score:.2f}. Display threshold set to: {display_threshold:.2f}"})
        else:
            self.update_llm_log_tabs({"type": "debug", "message": "No entities yet to calculate median score. Display threshold is default minimum."})

        filtered_for_display_tabs = [e for e in all_entities if e["current_importance_score"] >= display_threshold]
        filtered_for_display_tabs.sort(key=lambda e: (e["type"], -e["current_importance_score"], e["name"].lower()))

        def clear_layout(layout):
            if layout is None:
                return
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                nested_layout = item.layout()
                
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
                elif nested_layout is not None:
                    clear_layout(nested_layout)
                    del item 
                else: 
                    del item
        
        clear_layout(self.story_elements_container_layout)
        for entity in filtered_for_display_tabs:
            first_mentioned_time_seconds = entity.get("first_mentioned_idx", 0) * TRANSCRIPT_CHUNK_DURATION_SECONDS
            minutes = int(first_mentioned_time_seconds // 60)
            seconds = int(first_mentioned_time_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            entity_card = self._create_entity_card(entity, time_str) 
            self.story_elements_container_layout.addWidget(entity_card)
        self.story_elements_container_layout.addStretch() 

        clear_layout(self.key_characters_container_layout)
        key_characters = [e for e in filtered_for_display_tabs if e["type"] == "Characters"]
        key_characters.sort(key=lambda e: -e["current_importance_score"]) 

        for char_entity in key_characters:
            first_mentioned_time_seconds = char_entity.get("first_mentioned_idx", 0) * TRANSCRIPT_CHUNK_DURATION_SECONDS
            minutes = int(first_mentioned_time_seconds // 60)
            seconds = int(first_mentioned_time_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            char_card = self._create_entity_card(char_entity, time_str)
            self.key_characters_container_layout.addWidget(char_card)
        self.key_characters_container_layout.addStretch() 

        self.cheat_sheet_table.setRowCount(len(all_entities))
        
        for col_idx in range(self.cheat_sheet_table.columnCount()):
            if col_idx in self.cheat_sheet_column_widths:
                self.cheat_sheet_table.setColumnWidth(col_idx, self.cheat_sheet_column_widths[col_idx])

        all_entities_sorted_for_table = sorted(all_entities, key=lambda e: (e["type"], e["name"].lower()))

        for i, entity in enumerate(all_entities_sorted_for_table):
            self.cheat_sheet_table.setItem(i, 0, QTableWidgetItem(entity["name"]))
            self.cheat_sheet_table.setItem(i, 1, QTableWidgetItem(entity["type"]))
            description_item = QTableWidgetItem(entity.get("description", "No description available"))
            self.cheat_sheet_table.setItem(i, 2, description_item)
            self.cheat_sheet_table.setItem(i, 3, QTableWidgetItem(str(entity.get("base_importance_score", 0)))) 
            self.cheat_sheet_table.setItem(i, 4, QTableWidgetItem(str(entity.get("mention_count", 0))))      
            self.cheat_sheet_table.setItem(i, 5, QTableWidgetItem(str(entity.get("current_importance_score", 0)))) 
            
        char_count = sum(1 for e in filtered_for_display_tabs if e["type"] == "Characters")
        loc_count = sum(1 for e in filtered_for_display_tabs if e["type"] == "Locations")
        total_elements_displayed = len(filtered_for_display_tabs)

        self.characters_count_label.setText(str(char_count))
        self.locations_count_label.setText(str(loc_count))
        self.total_elements_count_label.setText(str(total_elements_displayed))
        
        self.recent_activity_display.append(f"Entities updated: {len(all_entities)} total found, {total_elements_displayed} displayed (>= threshold).")
        self.recent_activity_display.verticalScrollBar().setValue(self.recent_activity_display.verticalScrollBar().maximum())

    def update_llm_log_tabs(self, log_data):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        log_type = log_data.get("type", "unknown")
        message = log_data.get("message", "No message provided")
        data = log_data.get("data")

        formatted_message = f"<p style='margin-bottom: 5px;'>{timestamp} <b>[{log_type.upper()}]</b>: {message}</p>"
        
        if log_type == "prompt":
            formatted_message += f"<pre style='background-color: #e6e6fa; padding: 10px; border-radius: 5px;'><code>{json.dumps(data, indent=2)}</code></pre>"
        elif log_type == "raw_response":
            formatted_message += f"<pre style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><code>{data}</code></pre>"
        elif log_type == "parsed_entities":
            formatted_message += f"<pre style='background-color: #f0f8ff; padding: 10px; border-radius: 5px;'><code>{json.dumps(data, indent=2)}</code></pre>"
        elif log_type == "chat_prompt": 
            formatted_message += f"<pre style='background-color: #e0e7ff; padding: 10px; border-radius: 5px;'><code>{json.dumps(data, indent=2)}</code></pre>"
        elif log_type == "chat_response": 
            formatted_message += f"<pre style='background-color: #f0f2f5; padding: 10px; border-radius: 5px;'><code>{data}</code></pre>"
        elif log_type == "error":
             formatted_message = f"<p style='color: #CC0000; margin-bottom: 5px;'>{timestamp} <b>[ERROR]</b>: {message}</p>"
             if data: 
                 formatted_message += f"<pre style='background-color: #ffe6e6; padding: 10px; border-radius: 5px; color: #CC0000;'><code>{data}</code></pre>"
        elif log_type == "warning":
            formatted_message = f"<p style='color: #FF8C00; margin-bottom: 5px;'>{timestamp} <b>[WARNING]</b>: {message}</p>"
            if data: 
                formatted_message += f"<pre style='background-color: #fff8e6; padding: 10px; border-radius: 5px;'><code>{json.dumps(data, indent=2)}</code></pre>"
        elif log_type == "debug":
            formatted_message = f"<p style='color: #555555; margin-bottom: 5px;'>{timestamp} <b>[DEBUG]</b>: {message}</p>"
            if data:
                formatted_message += f"<pre style='background-color: #e0e0e0; padding: 10px; border-radius: 5px;'><code>{json.dumps(data, indent=2)}</code></pre>"
        elif log_type == "status": 
            formatted_message = f"<p style='color: #337ab7; margin-bottom: 5px;'>{timestamp} <b>[STATUS]</b>: {message}</p>"

        self.llm_raw_log_display.append(formatted_message)
        self.llm_raw_log_display.verticalScrollBar().setValue(self.llm_raw_log_display.verticalScrollBar().maximum())

        if log_type == "parsed_entities":
            self.llm_parsed_entities_table.setRowCount(len(data))
            for i, entity in enumerate(data):
                self.llm_parsed_entities_table.setItem(i, 0, QTableWidgetItem(entity.get("name", "")))
                self.llm_parsed_entities_table.setItem(i, 1, QTableWidgetItem(entity.get("type", "")))
                self.llm_parsed_entities_table.setItem(i, 2, QTableWidgetItem(entity.get("description", "")))
                self.llm_parsed_entities_table.setItem(i, 3, QTableWidgetItem(str(entity.get("base_importance_score", ""))))
            self.llm_parsed_entities_table.resizeColumnsToContents() 

        elif log_type in ["error", "warning", "chat_error"]:
            error_message = f"{timestamp} [{log_type.upper()}]: {message}"
            if "entity_data" in log_data and log_data["entity_data"]:
                error_message += f" (Entity: {json.dumps(log_data['entity_data'])})"
            self.llm_error_warnings_display.append(error_message)
            self.llm_error_warnings_display.verticalScrollBar().setValue(self.llm_error_warnings_display.verticalScrollBar().maximum())
        
    def send_chat_query(self):
        query = self.chat_input.text().strip()
        if not query:
            return
        self.chat_display.append(f"<div style='color: #333333; margin-bottom: 5px; font-weight: bold;'>User:</div><div style='background-color: #e0e7ff; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>{query}</div>")
        self.chat_thread.add_chat_query(query)
        self.chat_input.clear()
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def display_chat_response(self, response):
        self.chat_display.append(f"<div style='color: #6a0dad; margin-bottom: 5px; font-weight: bold;'>AI:</div><div style='background-color: #f0f2f5; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>{response}</div>")
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def closeEvent(self, event):
        try:
            self.stop_processing()
            if self.chat_thread:
                self.chat_thread.stop()
                self.chat_thread.wait()
            if self.web_search_thread and self.web_search_thread.isRunning():
                self.web_search_thread.quit()
                self.web_search_thread.wait()
            event.accept()
        except Exception as e:
            print(f"Error during cleanup: {e}", file=sys.stderr)
            event.accept()