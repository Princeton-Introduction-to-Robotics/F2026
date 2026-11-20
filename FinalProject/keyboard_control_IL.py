"""
Keyboard-controlled Crazyflie drone with data logging for imitation learning.
Captures: camera images, drone state estimates, and actions at each timestep.
Created for MAE345

Adapted from F2024-dev for the F2026 Crazyflie 2.1 Brushless + Skydroid 5.8G camera:
  - the drone is ARMED before takeoff (cf.platform.send_arming_request) -- without it
    the brushless motors silently never spin -- and disarmed after landing
  - PID controller + Kalman reset + wait-for-estimator before flying, as in the F2026 labs
  - recording starts automatically after the 3 s post-takeoff hover and stops when you
    land, before the descent; 'l' saves the demo, 'n' discards it
  - 'esc' = emergency stop (disarm + cut motors)
  - group / camera / crop / rate can be set on the command line (run with --help)

Run from a terminal (not Jupyter), with the mae345 env active:
    python keyboard_control_IL.py --group <radio channel> --camera <index>
"""

import argparse
import logging
import time
import cv2
import numpy as np
import json
import shutil
from datetime import datetime
from pathlib import Path
from threading import Thread
from pynput import keyboard

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander

# Configuration (defaults -- each can be overridden on the command line, see parse_args)
# Values below are from camera_test.ipynb (Parts 1, 3, 5).
group_number = 1  # Your drone's radio channel (--group)
CROP_SIZE = 80  # Rows cropped off the top of each frame (receiver overlay) (--crop)
camera_index = 0  # Skydroid receiver's index (--camera). macOS can renumber cameras when
                  # devices are plugged/unplugged -- the startup check below catches that.
RECEIVER_FRAME_SIZE = (640, 480)  # (width, height) the Skydroid receiver delivers
CONTROL_RATE_HZ = 20  # Adjust if needed
VELOCITY = 0.2
TAKEOFF_HEIGHT = 0.5
NUM_TRIALS = 99  # Highest trial number accepted (folders are trial_01 ... trial_99)
DATA_DIR = Path(__file__).resolve().parent / 'imitation_data'  # Always next to this script
URI = None  # Set in main() from group_number

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


class ImitationLearningDataCollector:
    """Collects state/action pairs with camera images for imitation learning."""
    
    def __init__(self, output_dir=DATA_DIR, trial_number=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session_dir = None
        self.saved_dir = None  # set once a demo is saved, for the analysis after landing
        self.data_log = []
        self.frame_count = 0
        self.start_time = None
        self.trial_number = trial_number
        
        # Current state and action - these get updated asynchronously
        self.current_state = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        self.current_action = [0.0, 0.0, 0.0, 0.0]  # [vx, vy, vz, yaw_rate]
        
        # For synchronized collection
        self.collecting = False
        self.image_shape = None
        self.last_capture_time = 0
        self.capture_period = 1.0 / CONTROL_RATE_HZ
        
        # For collection thread control
        self.thread_running = True  # separate flag for thread lifecycle
        
        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Make sure this is the receiver and not the Mac's built-in camera: check the
        # native frame size before forcing 640x480 below (the built-in one is 1920x1080).
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Camera {camera_index} opened but gave no frame -- is the receiver plugged in?")
        native_size = (frame.shape[1], frame.shape[0])
        if native_size != RECEIVER_FRAME_SIZE:
            self.cap.release()
            raise RuntimeError(
                f"Camera {camera_index} gives {native_size[0]}x{native_size[1]}, but the receiver gives "
                f"{RECEIVER_FRAME_SIZE[0]}x{RECEIVER_FRAME_SIZE[1]} -- this is probably the Mac's built-in camera. "
                f"Re-run the scan in camera_test.ipynb Part 1 and pass the receiver's index with --camera."
            )
        print(f"Camera {camera_index}: {native_size[0]}x{native_size[1]} (receiver)")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Data collector initialized")
        print(f"Collection rate: {CONTROL_RATE_HZ} Hz")
    
    def start_session(self):
        """Start a new data collection session."""
        if self.trial_number is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.output_dir / f"session_{timestamp}"
        else:
            self.session_dir = self.output_dir / f"trial_{self.trial_number:02d}"
        
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.session_dir / "images").mkdir(exist_ok=True)
        
        self.start_time = time.time()
        self.last_capture_time = self.start_time
        self.frame_count = 0
        self.data_log = []
        self.collecting = True
        
        print(f"Started data collection session: {self.session_dir}")
        print(f"Recording at {CONTROL_RATE_HZ} Hz (synchronized state-action-observation)")
    
    def update_state(self, state_dict):
        """Update the current state estimate from drone sensors."""
        self.current_state = state_dict.copy()
    
    def update_action(self, vx, vy, vz, yaw_rate):
        """Update the current action being sent to the drone."""
        self.current_action = [vx, vy, vz, yaw_rate]
    
    def try_capture_timestep(self):
        """
        Try to capture a timestep if enough time has elapsed.
        This ensures fixed-rate synchronized capture of state-action-observation.
        Returns True if captured, False otherwise.
        """
        if not self.collecting or self.session_dir is None:
            return False
        
        current_time = time.time()
        
        # Only capture if enough time has elapsed since last capture
        if current_time - self.last_capture_time < self.capture_period:
            return False
        
        # Capture camera frame
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Failed to capture frame")
            return False
        
        # Crop top portion to remove camera overlay (battery/frequency display)
        crop_top = CROP_SIZE  
        if frame.shape[0] > crop_top:
            frame = frame[crop_top:, :, :]
        self.image_shape = list(frame.shape)  # Saved image size (after crop), for metadata

        # Save image
        image_filename = f"frame_{self.frame_count:06d}.jpg"
        image_path = self.session_dir / "images" / image_filename
        cv2.imwrite(str(image_path), frame)
        
        # Create data entry with SYNCHRONIZED state, action, and observation
        data_entry = {
            'timestep': self.frame_count,
            'timestamp': current_time - self.start_time,
            'image_path': f"images/{image_filename}",
            'state': self.current_state.copy(),
            'action': self.current_action.copy(),
        }
        
        self.data_log.append(data_entry)
        
        # Print progress every 50 frames
        if self.frame_count % 50 == 0 and self.frame_count > 0:
            elapsed = current_time - self.start_time
            print(f"Collected {self.frame_count} timesteps ({elapsed:.1f}s, {self.frame_count/elapsed:.1f} Hz)")
        
        self.frame_count += 1
        self.last_capture_time = current_time
        
        return True
    
    def draw_overlay(self, frame):
        """Draw state and action information on the frame."""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1
        
        # Display trial number
        if self.trial_number is not None:
            cv2.putText(frame, f"Trial: {self.trial_number}", (10, y_offset), 
                        font, font_scale, color, thickness)
            y_offset += 20
        
        # Display frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, y_offset), 
                    font, font_scale, color, thickness)
        y_offset += 20
        
        # Display action
        action_str = f"Action: vx={self.current_action[0]:.2f}, vy={self.current_action[1]:.2f}, vz={self.current_action[2]:.2f}, yaw={self.current_action[3]:.2f}"
        cv2.putText(frame, action_str, (10, y_offset), 
                    font, font_scale, color, thickness)
        y_offset += 20
        
        # Display key state info if available
        if 'x' in self.current_state:
            pos_str = f"Pos: x={self.current_state['x']:.2f}, y={self.current_state['y']:.2f}, z={self.current_state['z']:.2f}"
            cv2.putText(frame, pos_str, (10, y_offset), 
                        font, font_scale, color, thickness)
    
    def save_session(self):
        """Save the collected data to JSON file."""
        self.collecting = False
        
        if self.session_dir is None or len(self.data_log) == 0:
            print("No data to save")
            return
        
        # Save main data log
        log_path = self.session_dir / "data_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.data_log, f, indent=2)
        
        # Save metadata
        metadata = {
            'session_name': self.session_dir.name,
            'trial_number': self.trial_number,
            'total_timesteps': self.frame_count,
            'duration': time.time() - self.start_time,
            'collection_rate_hz': 1.0 / self.capture_period,
            'camera_index': camera_index,
            'crop_top': CROP_SIZE,
            'image_shape': self.image_shape,  # [height, width, channels] after crop
            'state_variables': list(self.current_state.keys()),
            'action_format': ['vx', 'vy', 'vz', 'yaw_rate'],
            'data_format': 'Each entry contains synchronized state, action, and observation (image) at a fixed timestep'
        }
        
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.saved_dir = self.session_dir
        print(f"\nData saved to {self.session_dir}")
        print(f"Total timesteps collected: {self.frame_count}")
        print(f"Session duration: {metadata['duration']:.2f} seconds")
        print(f"Average collection rate: {self.frame_count / metadata['duration']:.1f} Hz")
    
    def discard_session(self):
        """Stop collecting and delete this session's folder (images included)."""
        self.collecting = False
        time.sleep(0.1)  # Let a capture already in progress finish writing its image

        if self.session_dir is not None and self.session_dir.exists():
            shutil.rmtree(self.session_dir)
            print(f"Deleted {self.session_dir} ({self.frame_count} timesteps)")
        self.data_log = []

    def stop_thread(self):
        """Signal the collection thread to stop."""
        self.thread_running = False
    
    def cleanup(self):
        """Release camera resources."""
        self.cap.release()


class KeyboardDroneWithLogging:
    """Keyboard control with integrated data logging."""
    
    def __init__(self, motion_commander, cf, data_collector):
        self.mc = motion_commander
        self.cf = cf  # Needed for arming/disarming and the emergency stop
        self.data_collector = data_collector
        self.is_flying = False
        self.is_recording = False
        
        # Movement parameters (m/s)
        self.velocity = VELOCITY
        
        # Current velocities
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0
        
        # Keys currently pressed
        self.pressed_keys = set()
        
        # Control thread
        self.running = False
        self.control_thread = None
        
        self.print_controls()
    
    def print_controls(self):
        print("\n=== Keyboard Controls ===")
        print("u: Take off (recording starts automatically after the 3 s hover)")
        print("w/s: Forward/Backward")
        print("a/d: Left/Right")
        print("space: Up")
        print("c: Down")
        print("q/e: Turn left/Right")
        print("l: Stop recording, SAVE, land and exit")
        print("n: Stop recording, DISCARD, land and exit")
        print("esc: EMERGENCY STOP (cut motors), save and exit")
        print("========================")
    
    def on_press(self, key):
        """Handle key press events."""
        try:
            # Handle character keys
            if hasattr(key, 'char'):
                if key.char == 'u' and not self.is_flying:
                    self.takeoff()
                elif key.char == 'l' and self.is_flying:
                    self.land(save=True)
                    return False  # Stop listener
                elif key.char == 'n' and self.is_flying:
                    self.land(save=False)
                    return False  # Stop listener
                elif key.char in ['l', 'n'] and not self.is_flying:
                    # Also confirms keys are reaching the script before takeoff
                    print(f"'{key.char}' received -- not flying yet, press 'u' to take off")
                elif key.char in ['w', 's', 'a', 'd', 'q', 'e', 'c']:
                    self.pressed_keys.add(key.char)
            
            # Handle special keys
            elif key == keyboard.Key.space:
                self.pressed_keys.add('space')
            elif key == keyboard.Key.esc:
                self.emergency_stop()
                return False  # Stop listener

        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release events."""
        try:
            if hasattr(key, 'char') and key.char in self.pressed_keys:
                self.pressed_keys.discard(key.char)
            elif key == keyboard.Key.space:
                self.pressed_keys.discard('space')
        except AttributeError:
            pass
    
    def takeoff(self):
        """Take off, hover, then start recording."""
        # Mandatory on the Crazyflie 2.1 Brushless: the motors will not spin
        # without this, and no error is raised if it's skipped. Drone must be
        # flat, still, and props clear when sent.
        print("Arming...")
        self.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        print("Taking off...")
        self.mc.take_off(TAKEOFF_HEIGHT, velocity=VELOCITY)
        self.is_flying = True
        
        # Wait for PID to stabilize
        print("Hovering for 3 seconds to stabilize PID...")
        time.sleep(3)
        
        print("PID stabilized - Ready to fly!")

        # Start control loop
        self.running = True
        self.control_thread = Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

        # Record every demo from the same stable hover
        self.start_recording()
        print("Press 'l' to save and land, 'n' to discard and land")

    def start_recording(self):
        """Start data collection."""
        if not self.is_flying:
            print("Cannot start recording - not flying!")
            return
        
        if self.is_recording:
            print("Already recording!")
            return
        
        print("▶ Recording STARTED")
        self.data_collector.start_session()
        self.is_recording = True
    
    def stop_recording(self):
        """Stop data collection."""
        if not self.is_recording:
            print("Not currently recording!")
            return
        
        print("⏹ Recording STOPPED")
        self.data_collector.save_session()
        self.is_recording = False
    
    def land(self, save=True):
        """Stop recording (save or discard it), then land."""
        # Stop recording BEFORE descending: the landing isn't part of the demo, and
        # its frames would be labeled with a hover action while the drone descends
        if self.is_recording:
            if save:
                self.stop_recording()
            else:
                print("✗ Recording DISCARDED")
                self.data_collector.discard_session()
                self.is_recording = False

        print("Landing...")
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        self.mc.land()
        self.is_flying = False

        # Disarm so the drone is left in a known-safe state
        self.cf.platform.send_arming_request(False)

    def emergency_stop(self):
        """Cut the motors immediately (the drone will drop), then save any data."""
        print("!!! EMERGENCY STOP - cutting motors !!!")
        self.running = False

        # Disarm first: a disarmed drone ignores any setpoints still being sent
        self.cf.platform.send_arming_request(False)
        self.cf.commander.send_stop_setpoint()

        # Stop the MotionCommander's background setpoint thread too (cflib has no
        # public call for this short of land(), which would fly a descent)
        if self.mc._thread is not None:
            self.mc._thread.stop()
            self.mc._thread = None
        self.mc._is_flying = False
        self.is_flying = False

        if self.is_recording:
            print("Saving data...")
            self.data_collector.save_session()
            self.is_recording = False

    def update_velocity_from_keys(self):
        """Update velocity commands based on currently pressed keys."""
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0
        
        if 'w' in self.pressed_keys:
            self.vx = self.velocity
        if 's' in self.pressed_keys:
            self.vx = -self.velocity
        if 'a' in self.pressed_keys:
            self.vy = self.velocity
        if 'd' in self.pressed_keys:
            self.vy = -self.velocity
        if 'space' in self.pressed_keys:
            self.vz = self.velocity
        if 'c' in self.pressed_keys:
            self.vz = -self.velocity
        if 'q' in self.pressed_keys:
            self.yaw_rate = 45  # degrees/sec
        if 'e' in self.pressed_keys:
            self.yaw_rate = -45
    
    def control_loop(self):
        """Main control loop - runs in separate thread."""
        rate = CONTROL_RATE_HZ
        dt = 1.0 / rate
        
        while self.running and self.is_flying:
            # Update velocities based on pressed keys
            self.update_velocity_from_keys()
            
            # Send command to drone
            self.mc.start_linear_motion(self.vx, self.vy, self.vz, self.yaw_rate)
            
            # Update action in data collector
            if self.is_recording:
                self.data_collector.update_action(self.vx, self.vy, self.vz, self.yaw_rate)
            
            time.sleep(dt)


# Determine initial position (same as the F2026 Lab 2 / Lab 5 flight cells):
def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001
    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


# Initialize controller (same as the F2026 labs):
def set_PID_controller(scf):
    print('Initializing PID Controller')
    scf.cf.param.set_value('stabilizer.controller', '1')
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(scf)
    time.sleep(0.1)
    print('Estimator converged')


def state_callback(timestamp, data, logconf, data_collector):
    """Callback to update state from drone logs."""
    # Update state variables as they come in
    for key, value in data.items():
        # Remove group prefix (e.g., 'stateEstimate.x' -> 'x')
        simple_key = key.split('.')[-1]
        data_collector.current_state[simple_key] = value


def collection_loop(data_collector):
    """
    Background thread that tries to capture data at fixed intervals.
    This ensures synchronized state-action-observation pairs.
    """
    print("Collection loop started")
    while data_collector.thread_running:  # FIXED: Use thread_running flag
        # Try to capture a timestep (will only succeed if collecting=True)
        data_collector.try_capture_timestep()
        
        # Small sleep to prevent busy waiting
        time.sleep(0.01)  # 10ms sleep, allows up to 100Hz checking
    
    print("Collection loop stopped")


def get_existing_trials(output_dir=DATA_DIR):
    """Get list of existing trial numbers."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    
    trials = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("trial_"):
            try:
                trial_num = int(item.name.split("_")[1])
                trials.append(trial_num)
            except (IndexError, ValueError):
                continue
    
    return sorted(trials)


def prompt_trial_number(output_dir=DATA_DIR, max_trials=NUM_TRIALS):
    """Prompt user for trial number with validation and overwrite checking."""
    existing_trials = get_existing_trials(output_dir)
    
    print("\n" + "="*50)
    print("TRIAL DATA COLLECTION")
    print("="*50)
    
    if existing_trials:
        print(f"\nExisting trials ({len(existing_trials)}): {existing_trials}")
        next_free = next(i for i in range(1, max_trials + 2) if i not in existing_trials)
        print(f"Next free trial number: {next_free}")
    else:
        print(f"\nNo existing trials found. Start with trial 1.")
    
    while True:
        try:
            trial_input = input(f"\nEnter trial number (1-{max_trials}): ").strip()
            trial_number = int(trial_input)
            
            if trial_number < 1 or trial_number > max_trials:
                print(f"Error: Trial number must be between 1 and {max_trials}")
                continue
            
            # Check if trial already exists
            trial_path = Path(output_dir) / f"trial_{trial_number:02d}"
            if trial_path.exists():
                print(f"\n⚠️  WARNING: Trial {trial_number} already exists!")
                print(f"Location: {trial_path}")
                
                # Check if it has data
                data_file = trial_path / "data_log.json"
                if data_file.exists():
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    print(f"Existing data: {len(data)} frames")
                
                overwrite = input("Do you want to OVERWRITE this trial? (yes/no): ").strip().lower()
                
                if overwrite in ['yes', 'y']:
                    # Remove existing data
                    shutil.rmtree(trial_path)
                    print(f"✓ Trial {trial_number} will be overwritten")
                    return trial_number
                else:
                    print("Please choose a different trial number.")
                    continue
            else:
                print(f"✓ Trial {trial_number} selected (new trial)")
                return trial_number
                
        except ValueError:
            print("Error: Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard-fly the Crazyflie and record IL trajectories.")
    parser.add_argument('--group', type=int, default=group_number,
                        help=f"Drone radio channel (default {group_number})")
    parser.add_argument('--camera', type=int, default=camera_index,
                        help=f"Video receiver index from camera_test.ipynb (default {camera_index})")
    parser.add_argument('--crop', type=int, default=CROP_SIZE,
                        help=f"Rows to crop off the top of each frame (default {CROP_SIZE})")
    parser.add_argument('--rate', type=float, default=CONTROL_RATE_HZ,
                        help=f"Control + recording rate in Hz (default {CONTROL_RATE_HZ})")
    parser.add_argument('--no-analysis', action='store_true',
                        help="Don't run analyze_il_data.py on the demo after saving it")
    parser.add_argument('--out', type=str, default=DATA_DIR.name,
                        help=f"Folder to save trials in, next to this script unless an absolute path "
                             f"(default {DATA_DIR.name})")
    args = parser.parse_args()
    if args.group is None:
        parser.error("set your drone's radio channel with --group (or group_number at the top of this file)")
    return args


def main():
    global group_number, camera_index, CROP_SIZE, CONTROL_RATE_HZ, URI, DATA_DIR
    args = parse_args()
    group_number = args.group
    camera_index = args.camera
    CROP_SIZE = args.crop
    CONTROL_RATE_HZ = args.rate
    DATA_DIR = Path(__file__).resolve().parent / args.out  # an absolute --out replaces the script folder
    URI = f'radio://0/{group_number}/2M'
    print(f"Drone: {URI} | camera index: {camera_index} | crop: {CROP_SIZE} rows | rate: {CONTROL_RATE_HZ} Hz")
    print(f"Saving to: {DATA_DIR}")

    # Prompt for trial number
    trial_number = prompt_trial_number(output_dir=DATA_DIR, max_trials=NUM_TRIALS)
    
    print("\nInitializing...")
    
    # Initialize data collector with trial number
    data_collector = ImitationLearningDataCollector(output_dir=DATA_DIR, trial_number=trial_number)
    
    # Initialize Crazyflie drivers
    cflib.crtp.init_drivers(enable_debug_driver=False)
    
    print("Connecting to Crazyflie...")
    
    # Start collection loop in background thread
    from threading import Thread
    collection_thread = Thread(target=collection_loop, args=(data_collector,), daemon=True)
    collection_thread.start()
    
    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            print("Connected!")

            # PID controller + Kalman reset, as in the F2026 labs. Drone must be
            # sitting flat and still on the ground while this runs.
            set_PID_controller(scf)

            print("Setting up logging...")
            
            # Set up logging configuration for state estimation
            # Split into two configs due to Crazyflie log size limits

            log_period_ms = int(1000 / CONTROL_RATE_HZ)  # Convert Hz to milliseconds
            
            # Position and velocity
            log_conf_pos = LogConfig(name='Position', period_in_ms=log_period_ms) 
            log_conf_pos.add_variable('stateEstimate.x', 'float')
            log_conf_pos.add_variable('stateEstimate.y', 'float')
            log_conf_pos.add_variable('stateEstimate.z', 'float')
            log_conf_pos.add_variable('stateEstimate.vx', 'float')
            log_conf_pos.add_variable('stateEstimate.vy', 'float')
            log_conf_pos.add_variable('stateEstimate.vz', 'float')
            
            # Orientation
            log_conf_att = LogConfig(name='Attitude', period_in_ms=log_period_ms)  
            log_conf_att.add_variable('stabilizer.roll', 'float')
            log_conf_att.add_variable('stabilizer.pitch', 'float')
            log_conf_att.add_variable('stabilizer.yaw', 'float')
            
            try:
                scf.cf.log.add_config(log_conf_pos)
                scf.cf.log.add_config(log_conf_att)
                print("Log configs added successfully")
            except AttributeError as e:
                print(f"Error adding log config: {e}")
                print("This might be due to invalid variable names or configuration size")
                raise
            
            # Set up callbacks to update state
            log_conf_pos.data_received_cb.add_callback(
                lambda ts, data, logconf: state_callback(ts, data, logconf, data_collector)
            )
            log_conf_att.data_received_cb.add_callback(
                lambda ts, data, logconf: state_callback(ts, data, logconf, data_collector)
            )
            
            log_conf_pos.start()
            log_conf_att.start()
            print("Logging started")
            
            # Create motion commander
            mc = MotionCommander(scf)
            
            # Create keyboard controller
            drone = KeyboardDroneWithLogging(mc, scf.cf, data_collector)

            print("\n" + "="*50)
            print("Ready to fly! Press 'u' to arm and take off.")
            print("Recording starts automatically after the 3 s hover.")
            print("'l' = save + land, 'n' = discard + land, 'esc' = emergency stop.")
            print(f"Data is collected at {CONTROL_RATE_HZ:g} Hz with synchronized")
            print("state-action-observation pairs.")
            print("="*50)
            
            # Start keyboard listener
            with keyboard.Listener(on_press=drone.on_press, on_release=drone.on_release) as listener:
                listener.join()
            
            # Stop logging
            log_conf_pos.stop()
            log_conf_att.stop()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop the collection thread
        data_collector.stop_thread()
        collection_thread.join(timeout=2.0)
        
        # Cleanup
        data_collector.cleanup()
        print("Done!")

    # Check the demo right away: plots + a video of what the camera recorded
    if data_collector.saved_dir is not None and not args.no_analysis:
        run_analysis(data_collector.saved_dir)


def run_analysis(trial_dir):
    """Run analyze_il_data.py on a just-saved demo and point to its outputs."""
    import matplotlib
    matplotlib.use('Agg')  # write the plots to files, no windows
    from analyze_il_data import analyze_trial

    print(f"\nAnalyzing {trial_dir} ...")
    try:
        analyze_trial(trial_dir)
    except Exception as e:
        print(f"Analysis failed ({e}); the demo itself is saved. "
              f"Re-run it with: python analyze_il_data.py {trial_dir}")
        return
    print(f"\nOutputs in {trial_dir}:")
    for name, what in [('trajectory.mp4', "video of the camera images (what the policy will see)"),
                       ('trajectory_analysis.png', 'flight path and state over time'),
                       ('action_analysis.png', 'the commands you sent'),
                       ('image_samples.png', 'sample frames: check there is no overlay text or static')]:
        print(f"  {name:24} {what}")
    print(f"Watch the video:  open {Path(trial_dir) / 'trajectory.mp4'}")


if __name__ == '__main__':
    main()
