#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// ---------- NETWORK CONFIGURATION ----------
#define WIFI_SSID "Name"
#define WIFI_PASSWORD "pass_word"

// ---------- FIREBASE CONFIGURATION ----------
#define API_KEY "your_API_key"
#define DATABASE_URL "Database_url"
#define USER_EMAIL "email@gmail.com"
#define USER_PASSWORD "Password"

// Firebase objects
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;
 
// ---------- MOTOR CONTROL PINS ----------
#define ENA 32
#define IN1 14
#define IN2 27
#define ENB 33
#define IN3 26
#define IN4 25

// ---------- LIMIT SWITCH PINS ----------
#define LIMIT_FRONT 4
#define LIMIT_BACK 5

#define PWM_SPEED 200  // PWM speed (0-255)

// ---------- SENSOR AND ACTUATOR PINS ----------
#define SERVO1_PIN 18       // First servo motor
#define SERVO2_PIN 19       // Second servo motor
#define RELAY_PIN 21        // Relay for water spray

// ---------- SYSTEM STATE VARIABLES ----------
enum RobotState {
  STATE_IDLE,
  STATE_CLEANING_FORWARD,
  STATE_CLEANING_BACKWARD,
  STATE_AT_FRONT,
  STATE_AT_BACK,
  STATE_HOMING,
  STATE_MANUAL_FORWARD,
  STATE_MANUAL_BACKWARD,
  STATE_ERROR
};

RobotState currentState = STATE_IDLE;
RobotState previousState = STATE_IDLE;

bool cleaning = false;
bool homing = false;
bool human_detected = false;
bool water_spray_active = false;
bool servos_activated = false;
String current_command = "stop";
unsigned long command_timestamp = 0;
unsigned long last_status_update = 0;
unsigned long last_command_check = 0;
unsigned long last_settings_check = 0;
unsigned long last_human_check = 0;

int position = 0;
unsigned long travel_start_time = 0;
unsigned long operation_start_time = 0;

// System settings
struct SystemSettings {
  unsigned long pause_ms = 2000;
  unsigned long travel_time_ms = 8000;
  int pwm_speed = 200;
};

SystemSettings settings;

bool firebase_initialized = false;
unsigned long last_firebase_attempt = 0;
const unsigned long FIREBASE_RETRY_INTERVAL = 10000;

// Servo objects
Servo servo1;
Servo servo2;

// Servo position variables
int servo1_home_position = 0;     // Home position (0°)
int servo1_active_position = 180; // Active position (180°)
int servo2_home_position = 0;     // Home position (0°)
int servo2_active_position = 180; // Active position (180°)

// ---------- FIREBASE FUNCTIONS ----------
void initializeFirebase() {
  Serial.println("Initializing Firebase...");
  
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  
  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  
  // Set database read/write timeout
  fbdo.setBSSLBufferSize(4096, 1024);
  fbdo.setResponseSize(2048);
  
  Serial.println("Firebase initialization started...");
}

bool checkFirebaseReady() {
  if (Firebase.ready()) {
    if (!firebase_initialized) {
      Serial.println("Firebase is now ready!");
      firebase_initialized = true;
      // Send initial status
      sendStatus("connected");
    }
    return true;
  } else {
    if (firebase_initialized) {
      Serial.println("❌ Firebase connection lost!");
      firebase_initialized = false;
    }
    return false;
  }
}

void sendStatus(const String& state) {
  if (millis() - last_status_update < 1000) { // Limit to 1 second intervals
    return;
  }
  
  if (!checkFirebaseReady()) {
    Serial.println("Firebase not ready, skipping status update");
    return;
  }

  FirebaseJson json;
  json.set("state", state);
  json.set("position", position);
  json.set("cleaning", cleaning);
  json.set("homing", homing);
  json.set("human_detected", human_detected);
  json.set("water_spray", water_spray_active);
  json.set("servos_active", servos_activated);
  
  // Read limit switches
  if (digitalRead(LIMIT_FRONT) == LOW) {
    json.set("last_switch", "front");
    if (state == "cleaning_forward" || state == "manual_forward") {
      position = 100; // At front position
    }
  } else if (digitalRead(LIMIT_BACK) == LOW) {
    json.set("last_switch", "back");
    if (state == "cleaning_backward" || state == "homing" || state == "manual_backward") {
      position = 0; // At home position
    }
  } else {
    json.set("last_switch", "none");
  }
  
  json.set("battery", 85); // Simulated battery level
  json.set("timestamp", millis());
  json.set("wifi_signal", WiFi.RSSI());

  Serial.print("Sending status: " + state + " | Position: " + String(position) + "%... ");
  
  if (Firebase.RTDB.setJSON(&fbdo, "/solar_cleaner/status", &json)) {
    Serial.println("OK");
  } else {
    Serial.println("FAILED: " + fbdo.errorReason());
  }
  
  last_status_update = millis();
}

void processCommand() {
  if (!checkFirebaseReady()) return;

  if (Firebase.RTDB.getJSON(&fbdo, "/solar_cleaner/command")) {
    FirebaseJson &json = fbdo.jsonObject();
    FirebaseJsonData result;
    
    if (json.get(result, "action")) {
      String new_command = result.stringValue;
      
      if (json.get(result, "timestamp")) {
        unsigned long new_timestamp = result.intValue;
        
        if (new_timestamp > command_timestamp) {
          command_timestamp = new_timestamp;
          current_command = new_command;
          Serial.println("New command received: " + current_command);
          
          // Execute command immediately
          executeCommand();
        }
      }
    }
  } else {
    Serial.println("Failed to read command: " + fbdo.errorReason());
  }
}

void executeCommand() {
  if (current_command == "start_clean" && !cleaning && !homing && !human_detected) {
    startCleaningCycle();
  } 
  else if (current_command == "stop") {
    stopAllOperations();
  } 
  else if (current_command == "go_home" && !cleaning && !homing) {
    goHome();
  }
  else if (current_command == "water_on" && !human_detected) {
    turnOnWaterSpray();
    current_command = "stop"; // Reset command after execution
  }
  else if (current_command == "water_off") {
    turnOffWaterSpray();
    current_command = "stop"; // Reset command after execution
  }
  else if (current_command == "forward" && !cleaning && !homing && !human_detected) {
    startManualMovement(true);
  }
  else if (current_command == "back" && !cleaning && !homing && !human_detected) {
    startManualMovement(false);
  }
}

void updateSettings() {
  if (!checkFirebaseReady()) return;

  if (Firebase.RTDB.getJSON(&fbdo, "/solar_cleaner/settings")) {
    FirebaseJson &json = fbdo.jsonObject();
    FirebaseJsonData result;
    
    if (json.get(result, "pause_ms")) {
      settings.pause_ms = result.intValue;
      Serial.println("⚙ Settings updated: pause_ms = " + String(settings.pause_ms));
    }
    
    if (json.get(result, "travel_time_ms")) {
      settings.travel_time_ms = result.intValue;
      Serial.println("⚙ Settings updated: travel_time_ms = " + String(settings.travel_time_ms));
    }
    
    if (json.get(result, "pwm_speed")) {
      settings.pwm_speed = result.intValue;
      Serial.println("⚙ Settings updated: pwm_speed = " + String(settings.pwm_speed));
    }
  }
}

void checkHumanDetection() {
  if (!checkFirebaseReady()) return;

  if (Firebase.RTDB.getBool(&fbdo, "/solar_cleaner/human_detected")) {
    bool new_human_detected = fbdo.boolData();
    
    if (new_human_detected != human_detected) {
      human_detected = new_human_detected;
      Serial.println("Human detection: " + String(human_detected ? "DETECTED" : "CLEAR"));
      
      if (human_detected) {
        // Emergency stop and return home
        emergencyStop();
      }
    }
  }
}

// ---------- COMPONENT FUNCTIONS ----------
void initializeServos() {
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  resetServos();
  Serial.println("Servos initialized");
}

void resetServos() {
  servo1.write(servo1_home_position);   // Home position (0°)
  servo2.write(servo2_home_position);   // Home position (0°)
  servos_activated = false;
  Serial.println("🔧 Servos reset to home position (0°)");
}

void activateServos() {
  servo1.write(servo1_active_position);  // Active position (180°)
  servo2.write(servo2_active_position);  // Active position (180°)
  servos_activated = true;
  Serial.println("Servos activated to 180° position");
}

void updateServoPosition() {
  // Check if robot is at the end (front) position
  if (checkFrontLimit()) {
    // At end position - activate servos to 180°
    if (!servos_activated) {
      activateServos();
      Serial.println("End position reached - Servos rotated to 180°");
    }
  } else {
    // Not at end position - return servos to home position (0°)
    if (servos_activated) {
      resetServos();
      Serial.println("Not at end position - Servos returned to 0°");
    }
  }
}

void turnOnWaterSpray() {
  digitalWrite(RELAY_PIN, HIGH);
  water_spray_active = true;
  Serial.println("Water spray ON");
  sendStatus("water_active");
}

void turnOffWaterSpray() {
  digitalWrite(RELAY_PIN, LOW);
  water_spray_active = false;
  Serial.println("Water spray OFF");
  sendStatus("idle");
}

// ---------- MOTOR FUNCTIONS ----------
void forward() {
  analogWrite(ENA, settings.pwm_speed);
  analogWrite(ENB, settings.pwm_speed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void backward() {
  analogWrite(ENA, settings.pwm_speed);
  analogWrite(ENB, settings.pwm_speed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

bool checkFrontLimit() {
  return digitalRead(LIMIT_FRONT) == LOW;
}

bool checkBackLimit() {
  return digitalRead(LIMIT_BACK) == LOW;
}

// ---------- OPERATION FUNCTIONS ----------
void startCleaningCycle() {
  if (human_detected) {
    Serial.println("Cleaning blocked - Human detected!");
    sendStatus("safety_lock");
    return;
  }
  
  cleaning = true;
  currentState = STATE_CLEANING_FORWARD;
  travel_start_time = millis();
  operation_start_time = millis();
  
  Serial.println("Starting cleaning cycle...");
  sendStatus("cleaning_forward");
  
  // Start forward movement with water spray
  turnOnWaterSpray();
  forward();
}

void continueCleaningCycle() {
  if (!cleaning) return;
  
  switch (currentState) {
    case STATE_CLEANING_FORWARD:
      if (checkFrontLimit()) {
        // Reached front limit
        stopMotors();
        turnOffWaterSpray();
        currentState = STATE_AT_FRONT;
        position = 100;
        
        // Update servo position based on limit switch
        updateServoPosition();
        
        Serial.println("Reached front position, pausing...");
        sendStatus("at_front");
        
        // Schedule backward movement after pause
        operation_start_time = millis();
      } else {
        // Continue forward
        forward();
        
        // Update servo position based on limit switch
        updateServoPosition();
        
        // Update position based on time
        updatePosition();
      }
      break;
      
    case STATE_AT_FRONT:
      if (millis() - operation_start_time >= settings.pause_ms) {
        // Start backward movement
        currentState = STATE_CLEANING_BACKWARD;
        travel_start_time = millis();
        Serial.println("Starting backward movement...");
        sendStatus("cleaning_backward");
        backward();
      }
      break;
      
    case STATE_CLEANING_BACKWARD:
      if (checkBackLimit()) {
        // Reached home position
        stopMotors();
        currentState = STATE_AT_BACK;
        position = 0;
        
        // Update servo position based on limit switch
        updateServoPosition();
        
        Serial.println("Reached home position, cleaning complete!");
        sendStatus("at_back");
        
        // Schedule completion
        operation_start_time = millis();
      } else {
        // Continue backward
        backward();
        
        // Update servo position based on limit switch
        updateServoPosition();
        
        // Update position based on time
        updatePosition();
      }
      break;
      
    case STATE_AT_BACK:
      if (millis() - operation_start_time >= settings.pause_ms) {
        // Cleaning cycle complete
        cleaning = false;
        currentState = STATE_IDLE;
        Serial.println("Cleaning cycle completed successfully!");
        sendStatus("idle");
      }
      break;
      
    default:
      break;
  }
}

void startManualMovement(bool isForward) {
  if (human_detected) {
    Serial.println("Manual movement blocked - Human detected!");
    return;
  }
  
  cleaning = false;
  homing = false;
  
  if (isForward) {
    currentState = STATE_MANUAL_FORWARD;
    Serial.println("⬆ Starting manual forward movement");
    sendStatus("manual_forward");
    forward();
  } else {
    currentState = STATE_MANUAL_BACKWARD;
    Serial.println("⬇ Starting manual backward movement");
    sendStatus("manual_backward");
    backward();
  }
}

void stopManualMovement() {
  if (currentState == STATE_MANUAL_FORWARD || currentState == STATE_MANUAL_BACKWARD) {
    stopMotors();
    currentState = STATE_IDLE;
    Serial.println("Manual movement stopped");
    sendStatus("idle");
  }
}

void goHome() {
  if (human_detected) {
    Serial.println("Homing blocked - Human detected!");
    return;
  }
  
  cleaning = false;
  homing = true;
  currentState = STATE_HOMING;
  
  Serial.println("Starting homing procedure...");
  sendStatus("homing");
  
  turnOffWaterSpray();
  backward();
}

void continueHoming() {
  if (!homing) return;
  
  if (checkBackLimit()) {
    // Reached home position
    stopMotors();
    homing = false;
    currentState = STATE_IDLE;
    position = 0;
    
    // Update servo position based on limit switch
    updateServoPosition();
    
    Serial.println("Home position reached");
    sendStatus("idle");
  } else {
    // Continue homing
    backward();
    
    // Update servo position based on limit switch
    updateServoPosition();
    
    // Update position during homing
    if (position > 0) {
      position = max(0, position - 2); // Gradually decrease position
    }
  }
}

void stopAllOperations() {
  Serial.println("Stopping all operations...");
  
  stopMotors();
  turnOffWaterSpray();
  
  cleaning = false;
  homing = false;
  currentState = STATE_IDLE;
  
  sendStatus("idle");
}

void emergencyStop() {
  Serial.println("EMERGENCY STOP - Human detected!");
  
  stopMotors();
  turnOffWaterSpray();
  resetServos();
  
  cleaning = false;
  currentState = STATE_IDLE;
  
  // Start emergency homing
  homing = true;
  goHome();
  
  sendStatus("emergency_stop");
}

void updatePosition() {
  if (cleaning && travel_start_time > 0) {
    unsigned long elapsed = millis() - travel_start_time;
    if (elapsed >= settings.travel_time_ms) {
      position = (currentState == STATE_CLEANING_FORWARD) ? 100 : 0;
    } else {
      int progress = (elapsed * 100) / settings.travel_time_ms;
      position = (currentState == STATE_CLEANING_FORWARD) ? progress : (100 - progress);
    }
    position = constrain(position, 0, 100);
  }
}

// ---------- SAFETY CHECKS ----------
void performSafetyChecks() {
  // Check limit switches during movement
  if (currentState == STATE_CLEANING_FORWARD || currentState == STATE_MANUAL_FORWARD) {
    if (checkFrontLimit()) {
      Serial.println("Front limit switch activated - stopping");
      stopMotors();
      if (cleaning) {
        currentState = STATE_AT_FRONT;
        position = 100;
        sendStatus("at_front");
      } else {
        stopManualMovement();
      }
    }
  }
  
  if (currentState == STATE_CLEANING_BACKWARD || currentState == STATE_MANUAL_BACKWARD || currentState == STATE_HOMING) {
    if (checkBackLimit()) {
      Serial.println("Back limit switch activated - stopping");
      stopMotors();
      if (cleaning) {
        currentState = STATE_AT_BACK;
        position = 0;
        sendStatus("at_back");
      } else if (homing) {
        homing = false;
        currentState = STATE_IDLE;
        sendStatus("idle");
      } else {
        stopManualMovement();
      }
    }
  }
  
  // Check for human detection during operations
  if (human_detected && (cleaning || homing || currentState == STATE_MANUAL_FORWARD || currentState == STATE_MANUAL_BACKWARD)) {
    emergencyStop();
  }
  
  // Update servo positions based on limit switch state
  updateServoPosition();
}

// ---------- SYSTEM MONITORING ----------
void printSystemStatus() {
  static unsigned long last_status_print = 0;
  if (millis() - last_status_print > 10000) { // Print every 10 seconds
    Serial.println("\n=== SYSTEM STATUS ===");
    Serial.println("State: " + String(currentState));
    Serial.println("Position: " + String(position) + "%");
    Serial.println("Cleaning: " + String(cleaning));
    Serial.println("Homing: " + String(homing));
    Serial.println("Human Detected: " + String(human_detected));
    Serial.println("Water Spray: " + String(water_spray_active));
    Serial.println("Servos Active: " + String(servos_activated));
    Serial.println("Front Limit: " + String(checkFrontLimit()));
    Serial.println("Back Limit: " + String(checkBackLimit()));
    Serial.println("Servo1 Position: " + String(servo1.read()) + "°");
    Serial.println("Servo2 Position: " + String(servo2.read()) + "°");
    Serial.println("WiFi RSSI: " + String(WiFi.RSSI()) + " dBm");
    Serial.println("====================\n");
    
    last_status_print = millis();
  }
}

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n" + String(80, '='));
  Serial.println("        TABLE CLEANING - ESP32 CONTROLLER");
  Serial.println("        Version 2.1 - Limit Switch Servo Control");
  Serial.println("        IR Sensor Removed - Using End Limit Switch Only");
  Serial.println(String(80, '='));

  // Initialize motor pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Initialize limit switches
  pinMode(LIMIT_FRONT, INPUT_PULLUP);
  pinMode(LIMIT_BACK, INPUT_PULLUP);

  // Initialize actuators
  pinMode(RELAY_PIN, OUTPUT);
  
  // Initialize servos
  initializeServos();
  
  // Ensure water spray is off
  turnOffWaterSpray();

  // Stop motors initially
  stopMotors();

  // WiFi connection
  Serial.print("📡 Connecting to WiFi: " + String(WIFI_SSID));
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  unsigned long wifiTimeout = millis() + 30000;
  while (WiFi.status() != WL_CONNECTED && millis() < wifiTimeout) {
    delay(500);
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n WiFi connected!");
    Serial.println("IP Address: " + WiFi.localIP().toString());
    Serial.println("Signal Strength: " + String(WiFi.RSSI()) + " dBm");
  } else {
    Serial.println("\n WiFi connection failed!");
    Serial.println("Please check WiFi credentials and restart...");
    while(1) {
      delay(5000);
    }
  }

  // Initialize Firebase
  initializeFirebase();

  // Wait a bit for Firebase to initialize
  delay(2000);

  // Initial settings update
  updateSettings();

  // Go to home position
  goHome();
  
  Serial.println("\n System initialization complete!");
  Serial.println(" Ready to receive commands from Python Control Panel");
  Serial.println(" Available commands: start_clean, stop, forward, back, go_home, water_on, water_off");
  Serial.println("  Servo Control: 0° at home, 180° at end position");
  Serial.println(String(80, '='));
}

// ---------- MAIN LOOP ----------
void loop() {
  // Maintain WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected! Attempting reconnect...");
    WiFi.reconnect();
    delay(5000);
    return;
  }

  // Check Firebase connection
  if (!checkFirebaseReady()) {
    if (millis() - last_firebase_attempt > FIREBASE_RETRY_INTERVAL) {
      Serial.println("Attempting Firebase reconnection...");
      initializeFirebase();
      last_firebase_attempt = millis();
    }
  }

  // Process Firebase commands (every 500ms)
  if (millis() - last_command_check > 500) {
    processCommand();
    last_command_check = millis();
  }

  // Check human detection status (every 2 seconds)
  if (millis() - last_human_check > 2000) {
    checkHumanDetection();
    last_human_check = millis();
  }
  
  // Update settings periodically (every 30 seconds)
  if (millis() - last_settings_check > 30000) {
    updateSettings();
    last_settings_check = millis();
  }
  
  // Perform ongoing operations
  if (cleaning) {
    continueCleaningCycle();
  } else if (homing) {
    continueHoming();
  }
  
  // Stop manual movement if command is stop
  if (current_command == "stop" && (currentState == STATE_MANUAL_FORWARD || currentState == STATE_MANUAL_BACKWARD)) {
    stopManualMovement();
    current_command = "";
  }
  
  // Perform safety checks and update servo positions
  performSafetyChecks();
  
  // Send status updates
  if (currentState != previousState) {
    switch (currentState) {
      case STATE_IDLE: sendStatus("idle"); break;
      case STATE_CLEANING_FORWARD: sendStatus("cleaning_forward"); break;
      case STATE_CLEANING_BACKWARD: sendStatus("cleaning_backward"); break;
      case STATE_AT_FRONT: sendStatus("at_front"); break;
      case STATE_AT_BACK: sendStatus("at_back"); break;
      case STATE_HOMING: sendStatus("homing"); break;
      case STATE_MANUAL_FORWARD: sendStatus("manual_forward"); break;
      case STATE_MANUAL_BACKWARD: sendStatus("manual_backward"); break;
      default: sendStatus("unknown"); break;
    }
    previousState = currentState;
  }
  
  // Print system status periodically
  printSystemStatus();
  
  delay(100); // Main loop delay
}