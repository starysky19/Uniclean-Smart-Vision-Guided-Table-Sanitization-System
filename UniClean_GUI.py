# Table_Cleaning_ai_control_panel
import sys
import time
import threading
import json
import requests
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path


from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QCheckBox, QLineEdit, QDateTimeEdit, QMessageBox,
    QProgressBar, QTextEdit, QGroupBox, QGridLayout,
    QSplitter, QFrame, QTabWidget
)
from PyQt5.QtCore import QTimer, Qt, QDateTime, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor

# ---------------- SYSTEM CONFIGURATION ----------------
FIREBASE_DATABASE_URL = "Database_url"
ESP32_CAMERA_STREAM_URL = "esp_cam_url"
FIREBASE_API_KEY = "Firebase_API_Key"

# YOLOv8 Model paths
YOLO_MODEL_PATH = "models/yolov8n.pt"  # Pre-trained model
CUSTOM_MODEL_PATH = "models/solar_panel_cleaner.pt"  #custom trained model

# Detection classes for cleaning
DIRT_CLASSES = ['dirt', 'dust', 'debris']
CLEAN_CLASS = 'clean'


# Real Firebase communication class
class FirebaseManager:
    def __init__(self):
        self.database_url = FIREBASE_DATABASE_URL
        self.api_key = FIREBASE_API_KEY
        self.last_status = {}
        self.connection_active = False

    def test_connection(self):
        """Test Firebase connection"""
        try:
            response = requests.get(f"{self.database_url}/solar_cleaner/status.json")
            return response.status_code == 200
        except:
            return False

    def send_command(self, command_action):
        """Send command to robot via Firebase"""
        try:
            command_data = {
                "action": command_action,
                "timestamp": int(time.time() * 1000)
            }

            response = requests.patch(
                f"{self.database_url}/solar_cleaner/command.json",
                data=json.dumps(command_data)
            )

            if response.status_code == 200:
                print(f"✅ Command sent: {command_action}")
                return True
            else:
                print(f"❌ Command failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Command error: {e}")
            return False

    def get_robot_status(self):
        """Get current robot status from Firebase"""
        try:
            response = requests.get(f"{self.database_url}/solar_cleaner/status.json")
            if response.status_code == 200:
                data = response.json()
                if data:
                    self.last_status = data
                    return data
            return self.last_status
        except Exception as e:
            print(f"❌ Status fetch error: {e}")
            return self.last_status

    def update_safety_status(self, human_detected):
        """Update human detection status"""
        try:
            response = requests.patch(
                f"{self.database_url}/solar_cleaner.json",
                data=json.dumps({"human_detected": human_detected})
            )
            return response.status_code == 200
        except:
            return False

    def update_system_settings(self, setting_key, setting_value):
        """Update robot settings"""
        try:
            settings_data = {setting_key: setting_value}
            response = requests.patch(
                f"{self.database_url}/solar_cleaner/settings.json",
                data=json.dumps(settings_data)
            )
            return response.status_code == 200
        except:
            return False


# AI Detection Manager
class AIDetectionManager:
    def __init__(self):
        self.model = None
        self.detection_enabled = False
        self.confidence_threshold = 0.5
        self.last_detection_time = 0
        self.detection_interval = 5  # seconds
        self.unclean_detected = False
        self.load_model()

    def load_model(self):
        """Load YOLOv8 model"""
        try:
            # Try to load custom model first
            if Path(CUSTOM_MODEL_PATH).exists():
                self.model = YOLO(CUSTOM_MODEL_PATH)
                print("✅ Custom YOLOv8 model loaded successfully")
            else:
                # Fallback to pre-trained model
                self.model = YOLO(YOLO_MODEL_PATH)
                print("✅ Pre-trained YOLOv8 model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading YOLOv8 model: {e}")
            self.model = None

    def detect_unclean(self, frame):
        """Detect unclean conditions in the frame"""
        if not self.model or not self.detection_enabled:
            return False, frame, 0.0

        # Limit detection frequency
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.unclean_detected, frame, 0.0

        try:
            # Perform detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            unclean_detected = False
            max_confidence = 0.0

            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = box.conf.item()
                        class_id = int(box.cls.item())
                        class_name = self.model.names[class_id]

                        # Check if detected object indicates unclean condition
                        if class_name.lower() in [dirt.lower() for dirt in DIRT_CLASSES]:
                            unclean_detected = True
                            max_confidence = max(max_confidence, confidence)

                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f'{class_name} {confidence:.2f}',
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 255), 2)

            self.last_detection_time = current_time
            self.unclean_detected = unclean_detected

            return unclean_detected, frame, max_confidence

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return False, frame, 0.0


# Create managers
firebase_manager = FirebaseManager()
ai_detector = AIDetectionManager()


# GUI event signals
class ControlSignals(QObject):
    robot_status_updated = pyqtSignal(dict)
    camera_frame_received = pyqtSignal(object)
    safety_alert_triggered = pyqtSignal(bool)
    connection_status_changed = pyqtSignal(bool)
    unclean_detected = pyqtSignal(bool, float)
    cleaning_triggered = pyqtSignal()
    gui_color_changed = pyqtSignal(str)  # 'clean', 'unclean', 'cleaning'


gui_signals = ControlSignals()


class ModernButton(QPushButton):
    """Custom modern button with hover effects"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(35)
        self.setCursor(Qt.PointingHandCursor)


class StatusIndicator(QLabel):
    """Custom status indicator with colors"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(25)
        self.setStyleSheet("""
            QLabel {
                border-radius: 6px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 10px;
            }
        """)

    def set_status(self, status_type, text):
        """Set status with color coding"""
        colors = {
            'success': '#10b981',
            'warning': '#f59e0b',
            'error': '#ef4444',
            'info': '#3b82f6',
            'idle': '#6b7280',
            'active': '#8b5cf6',
            'connected': '#10b981',
            'disconnected': '#ef4444',
            'clean': '#10b981',
            'unclean': '#f59e0b',
            'cleaning': '#3b82f6'
        }
        self.setText(text)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {colors.get(status_type, '#6b7280')};
                color: white;
                border-radius: 6px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 10px;
            }}
        """)


class WaterSprayControl(QWidget):
    """Water spray control widget with ON/OFF toggle"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.water_spray_active = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Header
        header = QLabel("💦 Water Spray Control")
        header.setStyleSheet("font-weight: bold; font-size: 11px; color: #e2e8f0;")
        layout.addWidget(header)

        # Control buttons
        button_layout = QHBoxLayout()

        self.water_on_button = ModernButton("🚰 Turn ON")
        self.water_off_button = ModernButton("💧 Turn OFF")

        self.water_on_button.clicked.connect(self.turn_on_water)
        self.water_off_button.clicked.connect(self.turn_off_water)

        # Style buttons
        self.water_on_button.setStyleSheet("""
            QPushButton {
                background: #10b981;
                color: white;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background: #34d399;
            }
            QPushButton:disabled {
                background: #6b7280;
            }
        """)

        self.water_off_button.setStyleSheet("""
            QPushButton {
                background: #ef4444;
                color: white;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background: #f87171;
            }
            QPushButton:disabled {
                background: #6b7280;
            }
        """)

        button_layout.addWidget(self.water_on_button)
        button_layout.addWidget(self.water_off_button)
        layout.addLayout(button_layout)

        # Status indicator
        self.water_status_indicator = StatusIndicator("OFF")
        self.water_status_indicator.set_status('idle', '💧 Water Spray: OFF')
        layout.addWidget(self.water_status_indicator)

    def turn_on_water(self):
        """Turn on water spray"""
        if not self.water_spray_active:
            if firebase_manager.send_command("water_on"):
                self.water_spray_active = True
                self.water_status_indicator.set_status('active', '🚰 Water Spray: ACTIVE')
                self.water_on_button.setEnabled(False)
                self.water_off_button.setEnabled(True)

    def turn_off_water(self):
        """Turn off water spray"""
        if self.water_spray_active:
            if firebase_manager.send_command("water_off"):
                self.water_spray_active = False
                self.water_status_indicator.set_status('idle', '💧 Water Spray: OFF')
                self.water_on_button.setEnabled(True)
                self.water_off_button.setEnabled(False)

    def update_status(self, water_active):
        """Update water spray status from robot"""
        self.water_spray_active = water_active
        if water_active:
            self.water_status_indicator.set_status('active', '🚰 Water Spray: ACTIVE')
            self.water_on_button.setEnabled(False)
            self.water_off_button.setEnabled(True)
        else:
            self.water_status_indicator.set_status('idle', '💧 Water Spray: OFF')
            self.water_on_button.setEnabled(True)
            self.water_off_button.setEnabled(False)


# ---------------- MAIN CONTROL PANEL ----------------
class UniCleanAIControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UniCleaning System - AI CONTROL")
        self.setMinimumSize(1200, 800)

        # System state
        self.robot_display_position = 0
        self.is_robot_moving = False
        self.human_detected = False
        self.pending_cleaning_operation = False
        self.connection_active = False
        self.current_gui_theme = 'clean'  # 'clean', 'unclean', 'cleaning'

        # AI Detection state
        self.ai_detection_active = False
        self.unclean_detected = False
        self.last_unclean_confidence = 0.0
        self.auto_clean_enabled = False

        # Water spray control
        self.water_spray_control = WaterSprayControl()

        # Camera state
        self.camera_capture = None
        self.detection_active = False

        # Water usage tracking
        self.water_usage_start_time = 0
        self.total_water_usage = 0

        self._setup_ui()
        self._setup_timers()
        self._render_robot_visualization()
        self._apply_gui_theme('clean')

        # Test connection on startup
        self.test_firebase_connection()

    def _setup_ui(self):
        """Setup the main user interface"""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create central splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel - Camera and Visualization (70%)
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Right Panel - Controls (30%)
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([800, 400])
        splitter.setHandleWidth(1)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_left_panel(self):
        """Create left panel with camera and visualization"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Header with connection status
        header_layout = QHBoxLayout()

        self.header_label = QLabel("UniCleaning System - AI VISION CONTROL")
        self.header_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: white;
                padding: 8px 0px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #059669);
                border-radius: 8px;
                text-align: center;
            }
        """)
        header_layout.addWidget(self.header_label)

        # Connection status indicator
        self.connection_indicator = StatusIndicator("Connecting...")
        self.connection_indicator.set_status('disconnected', '🔴 OFFLINE')
        header_layout.addWidget(self.connection_indicator)

        left_layout.addLayout(header_layout)

        # Camera Section with AI Detection
        camera_group = QGroupBox("🤖 AI Vision Monitoring & Detection")
        camera_group.setStyleSheet(self._get_groupbox_style())
        camera_layout = QVBoxLayout(camera_group)

        # AI Detection Status
        ai_status_layout = QHBoxLayout()
        self.ai_status_indicator = StatusIndicator("AI Ready")
        self.ai_status_indicator.set_status('info', '🤖 AI: Ready')

        self.detection_status_indicator = StatusIndicator("No Dirt Detected")
        self.detection_status_indicator.set_status('clean', '✅ Clean')

        ai_status_layout.addWidget(self.ai_status_indicator)
        ai_status_layout.addWidget(self.detection_status_indicator)
        camera_layout.addLayout(ai_status_layout)

        self.camera_display_label = QLabel()
        self.camera_display_label.setMinimumSize(640, 480)
        self.camera_display_label.setAlignment(Qt.AlignCenter)
        self.camera_display_label.setStyleSheet("""
            QLabel {
                background: #111827;
                border: 2px solid #374151;
                border-radius: 8px;
                color: #9ca3af;
                font-size: 12px;
            }
        """)
        self.camera_display_label.setText("AI Vision System\n\nClick 'Start AI Camera' for live detection")
        camera_layout.addWidget(self.camera_display_label)

        # Camera controls
        camera_controls_layout = QHBoxLayout()
        self.camera_start_button = ModernButton("🚀 Start AI Camera")
        self.camera_stop_button = ModernButton("⏹️ Stop Camera")
        self.auto_clean_toggle = ModernButton("🔴 Auto-Clean OFF")

        self.camera_start_button.clicked.connect(self.start_ai_camera)
        self.camera_stop_button.clicked.connect(self.stop_camera_monitoring)
        self.auto_clean_toggle.clicked.connect(self.toggle_auto_clean)

        camera_controls_layout.addWidget(self.camera_start_button)
        camera_controls_layout.addWidget(self.camera_stop_button)
        camera_controls_layout.addWidget(self.auto_clean_toggle)
        camera_layout.addLayout(camera_controls_layout)

        left_layout.addWidget(camera_group)

        # Status Indicators Row
        status_row = QHBoxLayout()

        # Safety Status
        safety_frame = QFrame()
        safety_frame.setStyleSheet("""
            QFrame {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        safety_layout = QVBoxLayout(safety_frame)
        safety_layout.addWidget(QLabel("🛡️ Safety"))
        self.safety_status_indicator = StatusIndicator("System Ready")
        self.safety_status_indicator.set_status('success', '✅ Secure')
        safety_layout.addWidget(self.safety_status_indicator)
        status_row.addWidget(safety_frame)

        # AI Detection Status
        ai_frame = QFrame()
        ai_frame.setStyleSheet("""
            QFrame {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        ai_layout = QVBoxLayout(ai_frame)
        ai_layout.addWidget(QLabel("🤖 AI Status"))
        self.ai_detection_indicator = StatusIndicator("Monitoring")
        self.ai_detection_indicator.set_status('info', '👁️ Monitoring')
        ai_layout.addWidget(self.ai_detection_indicator)
        status_row.addWidget(ai_frame)

        # Water Spray Status
        water_frame = QFrame()
        water_frame.setStyleSheet("""
            QFrame {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        water_layout = QVBoxLayout(water_frame)
        water_layout.addWidget(QLabel("💦 Water"))
        self.water_system_indicator = StatusIndicator("Ready")
        self.water_system_indicator.set_status('idle', '💧 Ready')
        water_layout.addWidget(self.water_system_indicator)
        status_row.addWidget(water_frame)

        left_layout.addLayout(status_row)

        # Robot Visualization
        viz_group = QGroupBox("📍 Robot Position Tracker")
        viz_group.setStyleSheet(self._get_groupbox_style())
        viz_layout = QVBoxLayout(viz_group)
        self.robot_visualization_label = QLabel()
        self.robot_visualization_label.setMinimumHeight(120)
        self.robot_visualization_label.setStyleSheet("""
            QLabel {
                background: #111827;
                border: 1px solid #374151;
                border-radius: 8px;
            }
        """)
        viz_layout.addWidget(self.robot_visualization_label)
        left_layout.addWidget(viz_group)

        return left_widget

    def _create_right_panel(self):
        """Create right panel with controls and status"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # System Status Card
        status_card = QFrame()
        status_card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e40af, stop:1 #1e3a8a);
                border-radius: 8px;
                padding: 10px;
            }
        """)
        status_layout = QVBoxLayout(status_card)

        status_header = QLabel("AI System Overview")
        status_header.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        status_layout.addWidget(status_header)

        self.system_status_label = QLabel("🤖 Robot: Connecting...")
        self.system_status_label.setStyleSheet("color: white; font-size: 11px; padding: 3px 0px;")
        status_layout.addWidget(self.system_status_label)

        # AI Detection Info
        self.ai_detection_label = QLabel("🔍 AI: No detection yet")
        self.ai_detection_label.setStyleSheet("color: white; font-size: 11px; padding: 3px 0px;")
        status_layout.addWidget(self.ai_detection_label)

        self.position_tracker = QProgressBar()
        self.position_tracker.setRange(0, 100)
        self.position_tracker.setValue(0)
        self.position_tracker.setFormat("Position: %p%")
        self.position_tracker.setStyleSheet("""
            QProgressBar {
                background: #1e3a8a;
                border: 1px solid #3b82f6;
                border-radius: 4px;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:0.5 #3b82f6, stop:1 #8b5cf6);
                border-radius: 3px;
            }
        """)
        status_layout.addWidget(self.position_tracker)

        # Connection controls
        connection_layout = QHBoxLayout()
        self.refresh_status_button = ModernButton("🔄 Refresh Status")
        self.test_connection_button = ModernButton("🔗 Test Connection")

        self.refresh_status_button.clicked.connect(self.update_robot_status)
        self.test_connection_button.clicked.connect(self.test_firebase_connection)

        connection_layout.addWidget(self.refresh_status_button)
        connection_layout.addWidget(self.test_connection_button)
        status_layout.addLayout(connection_layout)

        right_layout.addWidget(status_card)

        # Control Tabs
        self.control_tabs = QTabWidget()
        self.control_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #374151;
                border-radius: 6px;
                background: #1f2937;
            }
            QTabBar::tab {
                background: #374151;
                color: #d1d5db;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 10px;
            }
            QTabBar::tab:selected {
                background: #3b82f6;
                color: white;
            }
        """)

        # Tab 1: AI Control
        ai_tab = self._create_ai_control_tab()
        self.control_tabs.addTab(ai_tab, "🤖 AI Control")

        # Tab 2: Manual Control
        control_tab = self._create_control_tab()
        self.control_tabs.addTab(control_tab, "🎮 Manual")

        # Tab 3: Water Control
        water_tab = self._create_water_control_tab()
        self.control_tabs.addTab(water_tab, "💦 Water")

        # Tab 4: Settings
        settings_tab = self._create_settings_tab()
        self.control_tabs.addTab(settings_tab, "⚙️ Settings")

        right_layout.addWidget(self.control_tabs)

        # System Log
        log_group = QGroupBox("📝 AI System Log")
        log_group.setStyleSheet(self._get_groupbox_style())
        log_layout = QVBoxLayout(log_group)
        self.system_log = QTextEdit()
        self.system_log.setMaximumHeight(150)
        self.system_log.setReadOnly(True)
        self.system_log.setStyleSheet("""
            QTextEdit {
                background: #111827;
                border: 1px solid #374151;
                border-radius: 6px;
                color: #d1d5db;
                padding: 6px;
                font-family: 'Courier New';
                font-size: 9px;
            }
        """)
        log_layout.addWidget(self.system_log)
        right_layout.addWidget(log_group)

        return right_widget

    def _create_ai_control_tab(self):
        """Create AI control tab content"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        # AI Configuration
        ai_config_group = QGroupBox("⚙️ AI Configuration")
        ai_config_group.setStyleSheet(self._get_groupbox_style())
        ai_config_layout = QGridLayout(ai_config_group)

        ai_config_layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        self.confidence_slider_label = QLabel("0.5")
        self.confidence_slider = QProgressBar()
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setFormat("")

        ai_config_layout.addWidget(self.confidence_slider, 0, 1)
        ai_config_layout.addWidget(self.confidence_slider_label, 0, 2)

        ai_config_layout.addWidget(QLabel("Detection Interval (s):"), 1, 0)
        self.detection_interval_input = QLineEdit("5")
        self.detection_interval_input.setStyleSheet("""
            QLineEdit {
                background: #111827;
                border: 1px solid #374151;
                border-radius: 4px;
                padding: 6px;
                color: #d1d5db;
                font-size: 10px;
            }
        """)
        ai_config_layout.addWidget(self.detection_interval_input, 1, 1, 1, 2)

        self.save_ai_settings_button = ModernButton("💾 Save AI Settings")
        self.save_ai_settings_button.clicked.connect(self.save_ai_settings)
        ai_config_layout.addWidget(self.save_ai_settings_button, 2, 0, 1, 3)

        layout.addWidget(ai_config_group)

        # AI Detection Control
        ai_control_group = QGroupBox("🎯 AI Detection Control")
        ai_control_group.setStyleSheet(self._get_groupbox_style())
        ai_control_layout = QVBoxLayout(ai_control_group)

        self.enable_detection_checkbox = QCheckBox("Enable AI Detection")
        self.enable_detection_checkbox.setChecked(True)
        self.enable_detection_checkbox.stateChanged.connect(self.toggle_ai_detection)

        self.force_detection_button = ModernButton("🔍 Force Detection Now")
        self.force_detection_button.clicked.connect(self.force_detection)

        ai_control_layout.addWidget(self.enable_detection_checkbox)
        ai_control_layout.addWidget(self.force_detection_button)
        layout.addWidget(ai_control_group)

        # Auto-Clean Settings
        auto_clean_group = QGroupBox("🧹 Auto-Clean Settings")
        auto_clean_group.setStyleSheet(self._get_groupbox_style())
        auto_clean_layout = QVBoxLayout(auto_clean_group)

        self.auto_clean_checkbox = QCheckBox("Enable Auto-Clean Mode")
        self.auto_clean_checkbox.setChecked(False)
        self.auto_clean_checkbox.stateChanged.connect(self.toggle_auto_clean_mode)

        self.min_confidence_label = QLabel("Min Confidence for Auto-Clean: 0.7")
        self.min_confidence_slider = QProgressBar()
        self.min_confidence_slider.setRange(0, 100)
        self.min_confidence_slider.setValue(70)
        self.min_confidence_slider.setFormat("")

        auto_clean_layout.addWidget(self.auto_clean_checkbox)
        auto_clean_layout.addWidget(self.min_confidence_label)
        auto_clean_layout.addWidget(self.min_confidence_slider)
        layout.addWidget(auto_clean_group)

        layout.addStretch()
        return tab

    def _create_control_tab(self):
        """Create manual control tab content"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        # Manual Control
        manual_group = QGroupBox("👤 Manual Control")
        manual_group.setStyleSheet(self._get_groupbox_style())
        manual_layout = QGridLayout(manual_group)

        self.forward_button = ModernButton("⬆️ Forward")
        self.backward_button = ModernButton("⬇️ Backward")
        self.emergency_stop_button = ModernButton("🛑 Emergency Stop")

        self.emergency_stop_button.setStyleSheet("""
            QPushButton {
                background: #dc2626;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ef4444;
            }
        """)

        self.forward_button.clicked.connect(lambda: self._execute_manual_command("forward"))
        self.backward_button.clicked.connect(lambda: self._execute_manual_command("back"))
        self.emergency_stop_button.clicked.connect(lambda: self._execute_manual_command("stop"))

        manual_layout.addWidget(self.forward_button, 0, 0)
        manual_layout.addWidget(self.backward_button, 0, 1)
        manual_layout.addWidget(self.emergency_stop_button, 1, 0, 1, 2)

        layout.addWidget(manual_group)

        # Cleaning Operations
        cleaning_group = QGroupBox("🧹 Cleaning Operations")
        cleaning_group.setStyleSheet(self._get_groupbox_style())
        cleaning_layout = QVBoxLayout(cleaning_group)

        self.start_cleaning_button = ModernButton("Start Cleaning Cycle")
        self.return_home_button = ModernButton("🏠 Return Home")

        self.start_cleaning_button.clicked.connect(self._toggle_cleaning_operation)
        self.return_home_button.clicked.connect(lambda: self._send_robot_command("go_home"))

        cleaning_layout.addWidget(self.start_cleaning_button)
        cleaning_layout.addWidget(self.return_home_button)
        layout.addWidget(cleaning_group)

        # Safety Test
        safety_group = QGroupBox("🛡️ Safety Test")
        safety_group.setStyleSheet(self._get_groupbox_style())
        safety_layout = QHBoxLayout(safety_group)

        self.simulate_human_button = ModernButton("👤 Simulate Human")
        self.clear_human_button = ModernButton("✅ Clear Human")

        self.simulate_human_button.clicked.connect(lambda: self._simulate_human_detection(True))
        self.clear_human_button.clicked.connect(lambda: self._simulate_human_detection(False))

        safety_layout.addWidget(self.simulate_human_button)
        safety_layout.addWidget(self.clear_human_button)
        layout.addWidget(safety_group)

        layout.addStretch()
        return tab

    def _create_water_control_tab(self):
        """Create water control tab content"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        # Water Spray Control Group
        water_group = QGroupBox("💦 Water Spray Control")
        water_group.setStyleSheet(self._get_groupbox_style())
        water_layout = QVBoxLayout(water_group)

        # Add the water spray control widget
        water_layout.addWidget(self.water_spray_control)

        # Water usage statistics
        stats_group = QGroupBox("Water Usage Statistics")
        stats_group.setStyleSheet("""
            QGroupBox {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 8px;
                margin-top: 8px;
            }
        """)
        stats_layout = QVBoxLayout(stats_group)

        self.water_usage_label = QLabel("Total Usage: 0 minutes")
        self.water_usage_label.setStyleSheet("color: #d1d5db; font-size: 10px;")
        stats_layout.addWidget(self.water_usage_label)

        self.last_activation_label = QLabel("Last activated: Never")
        self.last_activation_label.setStyleSheet("color: #d1d5db; font-size: 10px;")
        stats_layout.addWidget(self.last_activation_label)

        water_layout.addWidget(stats_group)
        layout.addWidget(water_group)

        layout.addStretch()
        return tab

    def _create_settings_tab(self):
        """Create settings tab content"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        # Robot Configuration
        config_group = QGroupBox("⚙️ Robot Configuration")
        config_group.setStyleSheet(self._get_groupbox_style())
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel("Pause Duration (ms):"), 0, 0)
        self.pause_duration_input = QLineEdit("2000")
        self.pause_duration_input.setStyleSheet("""
            QLineEdit {
                background: #111827;
                border: 1px solid #374151;
                border-radius: 4px;
                padding: 6px;
                color: #d1d5db;
                font-size: 10px;
            }
        """)
        config_layout.addWidget(self.pause_duration_input, 0, 1)

        config_layout.addWidget(QLabel("Travel Time (ms):"), 1, 0)
        self.travel_time_input = QLineEdit("8000")
        self.travel_time_input.setStyleSheet("""
            QLineEdit {
                background: #111827;
                border: 1px solid #374151;
                border-radius: 4px;
                padding: 6px;
                color: #d1d5db;
                font-size: 10px;
            }
        """)
        config_layout.addWidget(self.travel_time_input, 1, 1)

        self.save_settings_button = ModernButton("💾 Save Settings")
        self.save_settings_button.clicked.connect(self.save_robot_settings)
        config_layout.addWidget(self.save_settings_button, 2, 0, 1, 2)

        layout.addWidget(config_group)

        # System Information
        info_group = QGroupBox("ℹ️ AI System Information")
        info_group.setStyleSheet(self._get_groupbox_style())
        info_layout = QVBoxLayout(info_group)

        info_text = QLabel(
            "UniCleaning System\n"
            "Version: 3.0 AI VISION\n"
            "Status: AI Detection Active\n"
            "Mode: YOLOv8 + ESP32 Camera\n\n"
            "Features:\n"
            "• Real-time object detection\n"
            "• Automatic cleaning trigger\n"
            "• Dynamic GUI theming\n"
            "• Safety monitoring"
        )
        info_text.setStyleSheet("color: #9ca3af; font-size: 10px; line-height: 1.4;")
        info_text.setAlignment(Qt.AlignCenter)
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        layout.addWidget(info_group)

        layout.addStretch()
        return tab

    def _get_groupbox_style(self):
        return """
        QGroupBox {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 10px;
            margin-top: 8px;
            font-weight: bold;
            color: #e2e8f0;
            font-size: 11px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0px 6px 0px 6px;
            color: #e2e8f0;
        }
        """

    def _setup_timers(self):
        """Setup timers for periodic updates"""
        # Set up application style
        self._apply_gui_theme('clean')

        # Camera frame timer
        self.camera_frame_timer = QTimer()
        self.camera_frame_timer.timeout.connect(self._capture_camera_frame)

        # Connect signals
        gui_signals.robot_status_updated.connect(self._handle_status_update)
        gui_signals.camera_frame_received.connect(self._display_camera_frame)
        gui_signals.safety_alert_triggered.connect(self._handle_safety_alert)
        gui_signals.connection_status_changed.connect(self._handle_connection_change)
        gui_signals.unclean_detected.connect(self._handle_unclean_detection)
        gui_signals.cleaning_triggered.connect(self._trigger_cleaning)
        gui_signals.gui_color_changed.connect(self._apply_gui_theme)

        # Status monitoring timer
        self.status_monitor_timer = QTimer()
        self.status_monitor_timer.timeout.connect(self.update_robot_status)
        self.status_monitor_timer.start(2000)  # Update every 2 seconds

        # Robot animation timer
        self.robot_animation_timer = QTimer()
        self.robot_animation_timer.timeout.connect(self._update_robot_visualization)

        # Water usage timer
        self.water_usage_timer = QTimer()
        self.water_usage_timer.timeout.connect(self._update_water_usage)

        # Connection monitoring timer
        self.connection_monitor_timer = QTimer()
        self.connection_monitor_timer.timeout.connect(self._monitor_connection)
        self.connection_monitor_timer.start(5000)  # Check every 5 seconds

        # AI Auto-clean check timer
        self.auto_clean_timer = QTimer()
        self.auto_clean_timer.timeout.connect(self._check_auto_clean)
        self.auto_clean_timer.start(1000)  # Check every second

    def _apply_gui_theme(self, theme):
        """Apply different GUI themes based on system state"""
        self.current_gui_theme = theme

        if theme == 'clean':
            # Green theme - everything is clean
            style = """
                QWidget {
                    background: #0f172a;
                    color: #e2e8f0;
                    font-family: 'Segoe UI';
                }
                QPushButton {
                    background: #10b981;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 12px;
                    font-weight: 600;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: #34d399;
                }
                QPushButton:pressed {
                    background: #059669;
                }
                QPushButton:disabled {
                    background: #4b5563;
                    color: #9ca3af;
                }
            """
            self.header_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: white;
                    padding: 8px 0px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #10b981, stop:1 #059669);
                    border-radius: 8px;
                    text-align: center;
                }
            """)

        elif theme == 'unclean':
            # Orange theme - unclean detected
            style = """
                QWidget {
                    background: #1c1917;
                    color: #e2e8f0;
                    font-family: 'Segoe UI';
                }
                QPushButton {
                    background: #f59e0b;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 12px;
                    font-weight: 600;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: #fbbf24;
                }
                QPushButton:pressed {
                    background: #d97706;
                }
                QPushButton:disabled {
                    background: #4b5563;
                    color: #9ca3af;
                }
            """
            self.header_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: white;
                    padding: 8px 0px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #f59e0b, stop:1 #d97706);
                    border-radius: 8px;
                    text-align: center;
                }
            """)

        elif theme == 'cleaning':
            # Blue theme - cleaning in progress
            style = """
                QWidget {
                    background: #0f1a2a;
                    color: #e2e8f0;
                    font-family: 'Segoe UI';
                }
                QPushButton {
                    background: #3b82f6;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 12px;
                    font-weight: 600;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: #60a5fa;
                }
                QPushButton:pressed {
                    background: #2563eb;
                }
                QPushButton:disabled {
                    background: #4b5563;
                    color: #9ca3af;
                }
            """
            self.header_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: white;
                    padding: 8px 0px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #3b82f6, stop:1 #1d4ed8);
                    border-radius: 8px;
                    text-align: center;
                }
            """)

        self.setStyleSheet(style)

    # ---------- CONNECTION MANAGEMENT ----------
    def test_firebase_connection(self):
        """Test Firebase connection"""
        self.log_system_event("Testing Firebase connection...")
        if firebase_manager.test_connection():
            self.connection_active = True
            gui_signals.connection_status_changed.emit(True)
            self.log_system_event("✅ Firebase connection successful!")
        else:
            self.connection_active = False
            gui_signals.connection_status_changed.emit(False)
            self.log_system_event("❌ Firebase connection failed!")

    def _monitor_connection(self):
        """Monitor connection status"""
        if not self.connection_active:
            self.test_firebase_connection()

    def _handle_connection_change(self, connected):
        """Handle connection status changes"""
        self.connection_active = connected
        if connected:
            self.connection_indicator.set_status('connected', '🟢 ONLINE')
            self.setWindowTitle("UniCleaning System - AI CONTROL [ONLINE]")
        else:
            self.connection_indicator.set_status('disconnected', '🔴 OFFLINE')
            self.setWindowTitle("UniCleaning System - AI CONTROL [OFFLINE]")

    # ---------- AI DETECTION FUNCTIONS ----------
    def start_ai_camera(self):
        """Start AI camera with object detection"""
        try:
            if self.camera_capture:
                self.log_system_event("Camera already active")
                return

            self.log_system_event("Starting AI camera with object detection...")

            # Try to connect to ESP32 camera
            self.camera_capture = cv2.VideoCapture(ESP32_CAMERA_STREAM_URL)

            if self.camera_capture.isOpened():
                self.camera_frame_timer.start(100)  # 10 FPS
                self.detection_active = True
                self.ai_detection_active = True
                self.log_system_event("✅ AI camera started successfully")
                self.ai_status_indicator.set_status('active', '🤖 AI: Active')
            else:
                self.camera_capture = None
                self.log_system_event("❌ Could not connect to ESP32 camera")
                # Fallback to simulated camera for testing
                self._start_simulated_camera()

        except Exception as e:
            self.log_system_event(f"❌ AI camera error: {str(e)}")
            self._start_simulated_camera()

    def _start_simulated_camera(self):
        """Start simulated camera for testing"""
        self.log_system_event("🔄 Starting simulated camera for testing...")
        self.camera_capture = "simulated"
        self.camera_frame_timer.start(100)
        self.detection_active = True
        self.ai_detection_active = True
        self.ai_status_indicator.set_status('active', '🤖 AI: Simulated')

    def toggle_ai_detection(self):
        """Toggle AI detection on/off"""
        self.ai_detection_active = self.enable_detection_checkbox.isChecked()
        if self.ai_detection_active:
            self.log_system_event("✅ AI detection enabled")
            self.ai_status_indicator.set_status('active', '🤖 AI: Active')
        else:
            self.log_system_event("⏸️ AI detection disabled")
            self.ai_status_indicator.set_status('idle', '🤖 AI: Idle')

    def force_detection(self):
        """Force immediate AI detection"""
        if self.camera_capture and self.ai_detection_active:
            self.log_system_event("🔍 Forcing immediate AI detection...")
            # Reset last detection time to force immediate detection
            ai_detector.last_detection_time = 0

    def toggle_auto_clean(self):
        """Toggle auto-clean mode"""
        self.auto_clean_enabled = not self.auto_clean_enabled
        if self.auto_clean_enabled:
            self.auto_clean_toggle.setText("🟢 Auto-Clean ON")
            self.auto_clean_toggle.setStyleSheet("background: #10b981; color: white;")
            self.log_system_event("✅ Auto-clean mode enabled")
        else:
            self.auto_clean_toggle.setText("🔴 Auto-Clean OFF")
            self.auto_clean_toggle.setStyleSheet("background: #ef4444; color: white;")
            self.log_system_event("⏸️ Auto-clean mode disabled")

    def toggle_auto_clean_mode(self):
        """Toggle auto-clean mode from checkbox"""
        self.auto_clean_enabled = self.auto_clean_checkbox.isChecked()

    def save_ai_settings(self):
        """Save AI configuration settings"""
        try:
            # Update confidence threshold
            confidence = self.confidence_slider.value() / 100.0
            ai_detector.confidence_threshold = confidence
            self.confidence_slider_label.setText(f"{confidence:.2f}")

            # Update detection interval
            interval = int(self.detection_interval_input.text())
            ai_detector.detection_interval = max(1, interval)

            self.log_system_event(f"⚙️ AI settings saved: Confidence={confidence:.2f}, Interval={interval}s")

        except Exception as e:
            self.log_system_event(f"❌ Error saving AI settings: {e}")

    def _check_auto_clean(self):
        """Check if auto-clean should be triggered"""
        if (self.auto_clean_enabled and self.unclean_detected and
                self.last_unclean_confidence > 0.7 and not self.human_detected):

            robot_status = firebase_manager.get_robot_status()
            if not robot_status.get("cleaning", False):
                self.log_system_event("🤖 Auto-clean triggered by AI detection!")
                self._trigger_cleaning()

    def _handle_unclean_detection(self, detected, confidence):
        """Handle unclean detection results"""
        self.unclean_detected = detected
        self.last_unclean_confidence = confidence

        if detected:
            self.detection_status_indicator.set_status('unclean', f'⚠️ Dirt Detected: {confidence:.2f}')
            self.ai_detection_label.setText(f"🔍 AI: Dirt detected (conf: {confidence:.2f})")

            if confidence > 0.5:
                gui_signals.gui_color_changed.emit('unclean')

        else:
            self.detection_status_indicator.set_status('clean', '✅ Clean')
            self.ai_detection_label.setText("🔍 AI: No dirt detected")
            gui_signals.gui_color_changed.emit('clean')

    def _trigger_cleaning(self):
        """Trigger cleaning operation"""
        if not self.human_detected:
            self.log_system_event("🧹 Starting cleaning cycle triggered by AI")
            self._send_robot_command("start_clean")
            gui_signals.gui_color_changed.emit('cleaning')

    # ---------- CAMERA SYSTEM ----------
    def _capture_camera_frame(self):
        """Capture camera frame and perform AI detection"""
        try:
            if not self.camera_capture:
                return

            if self.camera_capture == "simulated":
                # Create simulated frame
                frame = self._create_simulated_frame()
            else:
                # Capture from real camera
                ret, frame = self.camera_capture.read()
                if not ret:
                    self.log_system_event("❌ Camera frame capture failed")
                    self.stop_camera_monitoring()
                    return

            # Perform AI detection if enabled
            if self.ai_detection_active:
                unclean_detected, processed_frame, confidence = ai_detector.detect_unclean(frame)

                # Emit detection results
                if unclean_detected != self.unclean_detected or confidence != self.last_unclean_confidence:
                    gui_signals.unclean_detected.emit(unclean_detected, confidence)

                frame = processed_frame
            else:
                # Just display frame without detection
                cv2.putText(frame, "AI Detection Disabled", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            gui_signals.camera_frame_received.emit(frame)

        except Exception as e:
            self.log_system_event(f"Camera capture error: {e}")

    def _create_simulated_frame(self):
        """Create simulated camera frame for testing"""
        width, height = 640, 480
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Add some visual elements
        cv2.putText(frame, "TABLE CLEANING SYSTEM - SIMULATED VIEW", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Simulate occasional dirt detection
        if int(time.time()) % 10 == 0:
            cv2.putText(frame, "SIMULATED DIRT DETECTED", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Draw a simulated dirt patch
            cv2.rectangle(frame, (200, 200), (400, 300), (0, 0, 255), -1)
            cv2.putText(frame, "DIRT", (250, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame

    def _display_camera_frame(self, frame_data):
        """Display camera frame with AI annotations"""
        try:
            # Resize frame for display
            frame = cv2.resize(frame_data, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_display_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            print(f"Display error: {e}")

    def stop_camera_monitoring(self):
        """Stop camera monitoring"""
        try:
            if self.camera_capture:
                self.camera_frame_timer.stop()
                if hasattr(self.camera_capture, 'release'):
                    self.camera_capture.release()
                self.camera_capture = None
                self.detection_active = False
                self.ai_detection_active = False
                self.camera_display_label.clear()
                self.camera_display_label.setText("AI Vision System\n\nClick 'Start AI Camera' for live detection")
                self.log_system_event("Camera stopped")
                self.ai_status_indicator.set_status('idle', '🤖 AI: Idle')
        except Exception as e:
            self.log_system_event(f"Error stopping camera: {str(e)}")

    # ---------- COMMAND FUNCTIONS ----------
    def _send_robot_command(self, command):
        """Send command to robot"""
        if not self.connection_active:
            self.log_system_event("❌ Cannot send command - No connection")
            return False

        success = firebase_manager.send_command(command)
        if success:
            self.log_system_event(f"📤 Command sent: {command}")
        else:
            self.log_system_event(f"❌ Command failed: {command}")
        return success

    def _execute_manual_command(self, control_action):
        """Execute manual command"""
        if self.human_detected:
            self.log_system_event("❌ Manual control blocked - Safety alert active!")
            return

        self._send_robot_command(control_action)

    def _toggle_cleaning_operation(self):
        """Toggle cleaning operation"""
        if self.human_detected:
            self.log_system_event("❌ Cleaning blocked - Safety alert active!")
            return

        robot_status = firebase_manager.get_robot_status()
        if robot_status.get("cleaning", False):
            self._send_robot_command("stop")
            self.start_cleaning_button.setText("Start Cleaning Cycle")
        else:
            self._send_robot_command("start_clean")
            self.start_cleaning_button.setText("⏹️ Stop Cleaning")

    # ---------- SAFETY AND DETECTION SYSTEM ----------
    def _simulate_human_detection(self, detected):
        """Simulate human detection for testing"""
        self.human_detected = detected
        if firebase_manager.update_safety_status(detected):
            gui_signals.safety_alert_triggered.emit(detected)

            if detected:
                self.log_system_event("🚫 SIMULATION: Human detected - Safety system activated!")
                self.safety_status_indicator.set_status('error', '🚫 HUMAN')
            else:
                self.log_system_event("✅ SIMULATION: Human cleared - Area secure")
                self.safety_status_indicator.set_status('success', '✅ Secure')
        else:
            self.log_system_event("❌ Failed to update safety status")

    def _handle_safety_alert(self, alert_active):
        """Handle safety alerts"""
        if alert_active:
            self.start_cleaning_button.setEnabled(False)
            self.start_cleaning_button.setText("❌ Safety Lock")
            self.forward_button.setEnabled(False)
            self.backward_button.setEnabled(False)
            self.water_spray_control.water_on_button.setEnabled(False)

            # Auto-stop water if active
            if self.water_spray_control.water_spray_active:
                self.water_spray_control.turn_off_water()
        else:
            self.start_cleaning_button.setEnabled(True)
            self.start_cleaning_button.setText("Start Cleaning Cycle")
            self.forward_button.setEnabled(True)
            self.backward_button.setEnabled(True)
            if not self.water_spray_control.water_spray_active:
                self.water_spray_control.water_on_button.setEnabled(True)

    # ---------- WATER SPRAY CONTROL FUNCTIONS ----------
    def _update_water_usage(self):
        """Update water usage statistics"""
        if self.water_spray_control.water_spray_active:
            current_usage = int((time.time() - self.water_usage_start_time) / 60)
            self.water_usage_label.setText(f"Total Usage: {self.total_water_usage + current_usage} minutes")

    def _handle_water_spray_change(self, active):
        """Handle water spray state changes"""
        if active:
            # Water turned on
            self.water_usage_start_time = time.time()
            self.water_usage_timer.start(60000)  # Update every minute
            self.last_activation_label.setText(f"Last activated: {datetime.now().strftime('%H:%M:%S')}")
            self.log_system_event("🚰 Water spray activated")
            self.water_system_indicator.set_status('active', '🚰 Water Active')
        else:
            # Water turned off
            if self.water_usage_start_time > 0:
                session_usage = int((time.time() - self.water_usage_start_time) / 60)
                self.total_water_usage += session_usage
                self.water_usage_start_time = 0
            self.water_usage_timer.stop()
            self.log_system_event("💧 Water spray deactivated")
            self.water_system_indicator.set_status('idle', '💧 System Ready')

    # ---------- ROBOT VISUALIZATION SYSTEM ----------
    def _update_robot_visualization(self):
        """Update robot visualization"""
        if not self.is_robot_moving:
            return

        # Smooth animation toward target position
        if self.robot_display_position < self.actual_robot_position:
            self.robot_display_position = min(self.actual_robot_position, self.robot_display_position + 3)
        elif self.robot_display_position > self.actual_robot_position:
            self.robot_display_position = max(self.actual_robot_position, self.robot_display_position - 3)

        if self.robot_display_position == self.actual_robot_position:
            self.is_robot_moving = False
            self.robot_animation_timer.stop()

        self._render_robot_visualization()

    def _render_robot_visualization(self):
        """Render robot visualization"""
        try:
            visualization_image = np.zeros((120, 600, 3), dtype=np.uint8)

            # Draw track
            cv2.rectangle(visualization_image, (10, 40), (590, 60), (60, 60, 60), -1)
            cv2.rectangle(visualization_image, (10, 40), (590, 60), (100, 100, 100), 1)

            # Draw position markers
            cv2.putText(visualization_image, "HOME", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(visualization_image, "END", (575, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Draw robot
            robot_color = (0, 220, 220)  # Cyan
            if self.is_robot_moving:
                robot_color = (0, 255, 150)  # Brighter when moving

            cv2.rectangle(visualization_image, (self.robot_display_position, 30),
                          (self.robot_display_position + 25, 70), robot_color, -1)
            cv2.rectangle(visualization_image, (self.robot_display_position, 30),
                          (self.robot_display_position + 25, 70), (255, 255, 255), 1)
            cv2.putText(visualization_image, "ROBOT", (self.robot_display_position + 2, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

            # Get current status for visualization
            robot_status = firebase_manager.get_robot_status()
            water_spray_active = robot_status.get('water_spray', False)
            servos_active = robot_status.get('servos_active', False)
            operation_state = robot_status.get('state', 'idle')

            # Visualize water spray
            if water_spray_active:
                spray_x = self.robot_display_position + 25
                for i in range(3):
                    y_pos = 25 - i * 4
                    cv2.line(visualization_image, (spray_x, 50), (spray_x + 20, y_pos), (0, 100, 255), 2)
                cv2.putText(visualization_image, "WATER", (spray_x + 2, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 255), 1)

            # Visualize servo activation
            if servos_active:
                cv2.circle(visualization_image, (self.robot_display_position + 6, 25), 4, (255, 100, 0), -1)
                cv2.circle(visualization_image, (self.robot_display_position + 19, 25), 4, (255, 100, 0), -1)

            # Display operation status
            status_text = f"State: {operation_state}"
            cv2.putText(visualization_image, status_text, (10, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # AI Detection status
            ai_status = "AI: Clean" if not self.unclean_detected else f"AI: Dirt ({self.last_unclean_confidence:.2f})"
            cv2.putText(visualization_image, ai_status, (400, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Convert and display
            rgb_image = cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, 600, 120, 600 * 3, QImage.Format_RGB888)
            self.robot_visualization_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            print(f"Visualization error: {e}")

    def _update_robot_movement_display(self, status_data):
        """Update robot movement display"""
        position_percentage = status_data.get('position', 0)
        cleaning_active = status_data.get('cleaning', False)
        homing_active = status_data.get('homing', False)

        # Calculate pixel position from percentage
        target_pixel_position = int((position_percentage / 100.0) * 565)  # 590-25 for robot width
        self.actual_robot_position = target_pixel_position

        # Determine movement state
        previous_movement_state = self.is_robot_moving
        self.is_robot_moving = cleaning_active or homing_active or "manual" in status_data.get('state', '')

        # Control animation timer
        if self.is_robot_moving and not self.robot_animation_timer.isActive():
            self.robot_animation_timer.start(50)  # 20 FPS
        elif not self.is_robot_moving and self.robot_animation_timer.isActive():
            self.robot_animation_timer.stop()
            self.robot_display_position = target_pixel_position
            self._render_robot_visualization()

        # Redraw if movement state changed
        if previous_movement_state != self.is_robot_moving:
            self._render_robot_visualization()

    # ---------- STATUS MONITORING ----------
    def update_robot_status(self):
        """Update robot status"""
        try:
            if not self.connection_active:
                return

            current_status = firebase_manager.get_robot_status()
            if current_status:
                gui_signals.robot_status_updated.emit(current_status)
        except Exception as e:
            self.log_system_event(f"Status error: {str(e)}")

    def _handle_status_update(self, status_data):
        """Handle status update"""
        operation_state = status_data.get('state', 'unknown')
        position_percentage = status_data.get('position', 0)
        battery_level = status_data.get('battery', 'n/a')
        water_spray_active = status_data.get('water_spray', False)
        human_detected = status_data.get('human_detected', False)

        # Update water spray status
        self.water_spray_control.update_status(water_spray_active)
        self._handle_water_spray_change(water_spray_active)

        # Update human detection
        if human_detected != self.human_detected:
            self.human_detected = human_detected
            gui_signals.safety_alert_triggered.emit(human_detected)

        # Update status display
        status_icon = "🤖"
        status_type = 'info'

        if "clean" in operation_state.lower():
            status_icon = "🧹"
            status_type = 'success'
        elif "home" in operation_state.lower():
            status_icon = "🏠"
            status_type = 'warning'
        elif "error" in operation_state.lower():
            status_icon = "❌"
            status_type = 'error'
        elif "idle" in operation_state.lower():
            status_icon = "⏸️"
            status_type = 'idle'

        self.system_status_label.setText(
            f"{status_icon} {operation_state} | 🔋 {battery_level}% | 📍 {position_percentage}%")

        self.position_tracker.setValue(position_percentage)

        # Update robot visualization
        self._update_robot_movement_display(status_data)

        # Update cleaning button state
        if status_data.get("cleaning", False):
            self.start_cleaning_button.setText("⏹️ Stop Cleaning")
            self.start_cleaning_button.setStyleSheet("background: #dc2626; color: white; font-weight: bold;")
            gui_signals.gui_color_changed.emit('cleaning')
        else:
            self.start_cleaning_button.setText("Start Cleaning Cycle")
            self.start_cleaning_button.setStyleSheet("")
            # Only change to clean theme if not unclean detected
            if not self.unclean_detected:
                gui_signals.gui_color_changed.emit('clean')

    # ---------- SETTINGS FUNCTIONS ----------
    def save_robot_settings(self):
        """Save robot configuration settings"""
        try:
            pause_setting = int(self.pause_duration_input.text())
            travel_setting = int(self.travel_time_input.text())

            if pause_setting < 500 or travel_setting < 1000:
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "Pause duration must be ≥500ms, Travel time must be ≥1000ms"
                )
                return

            if (firebase_manager.update_system_settings("pause_ms", pause_setting) and
                    firebase_manager.update_system_settings("travel_time_ms", travel_setting)):
                self.log_system_event(
                    f"⚙️ Settings saved: Pause={pause_setting}ms, Travel={travel_setting}ms"
                )
            else:
                self.log_system_event("❌ Failed to save settings")
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers")

    def log_system_event(self, event_message):
        """Log system event"""
        try:
            event_timestamp = datetime.now().strftime('%H:%M:%S')
            self.system_log.append(f"[{event_timestamp}] {event_message}")
            # Auto-scroll to bottom
            self.system_log.verticalScrollBar().setValue(
                self.system_log.verticalScrollBar().maximum()
            )
        except Exception as e:
            print(f"Log error: {e}")

    def closeEvent(self, close_event):
        """Handle application close"""
        try:
            self.stop_camera_monitoring()
            self.robot_animation_timer.stop()
            self.status_monitor_timer.stop()
            self.water_usage_timer.stop()
            self.connection_monitor_timer.stop()
            self.auto_clean_timer.stop()
        except Exception as e:
            print(f"Close error: {e}")
        close_event.accept()


# ---------- APPLICATION ENTRY POINT ----------
if __name__ == "__main__":
    try:
        # Set up application
        control_panel_app = QApplication(sys.argv)
        control_panel_app.setStyle('Fusion')

        # Create and show main window
        main_control_panel = UniCleanAIControlPanel()
        main_control_panel.show()

        print("🚀 Table Cleaning System - AI VISION STARTED")
        print("📋 AI Features:")
        print("   • YOLOv8 Object Detection")
        print("   • ESP32 Camera Integration")
        print("   • Automatic Cleaning Trigger")
        print("   • Dynamic GUI Color Themes")
        print("   • Real-time Dirt Detection")
        print("   • Auto-clean Mode")
        print("   • Confidence-based Decisions")

        # Start application
        exit_code = control_panel_app.exec_()
        sys.exit(exit_code)

    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)