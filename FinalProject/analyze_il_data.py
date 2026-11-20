"""
Process and analyze imitation learning data collected from Crazyflie drone.
Validates data quality and prepares it for training an IL policy.
"""

import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image


class ImitationDataProcessor:
    """Process and validate collected imitation learning data."""
    
    def __init__(self, trial_dir):
        """
        Initialize processor for a specific trial.
        
        Args:
            trial_dir: Path to trial directory (e.g., 'imitation_data/trial_01')
        """
        self.trial_dir = Path(trial_dir)
        if not self.trial_dir.exists():
            raise ValueError(f"Trial directory not found: {trial_dir}")
        
        self.data_log = None
        self.metadata = None
        self.images_dir = self.trial_dir / "images"
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load data log and metadata."""
        # Load data log
        data_log_path = self.trial_dir / "data_log.json"
        if not data_log_path.exists():
            raise ValueError(f"data_log.json not found in {self.trial_dir}")
        
        with open(data_log_path, 'r') as f:
            self.data_log = json.load(f)
        
        # Load metadata
        metadata_path = self.trial_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"Loaded trial: {self.trial_dir.name}")
        print(f"Total timesteps: {len(self.data_log)}")
        if self.metadata:
            print(f"Duration: {self.metadata['duration']:.2f} seconds")
            print(f"Collection rate: {self.metadata['collection_rate_hz']} Hz")
    
    def validate_data(self):
        """
        Validate data integrity and completeness.
        
        Returns:
            dict: Validation results
        """
        print("\n" + "="*60)
        print("DATA VALIDATION")
        print("="*60)
        
        validation = {
            'total_timesteps': len(self.data_log),
            'missing_images': 0,
            'image_shape': None,
            'state_dims': {},
            'action_dims': 4,
            'temporal_gaps': [],
            'issues': []
        }
        
        # Gap threshold scales with the collection rate (1.5x the expected period)
        rate_hz = self.metadata['collection_rate_hz'] if self.metadata else 10
        max_gap = 1.5 / rate_hz
        
        # Check each timestep
        for i, entry in enumerate(self.data_log):
            # Validate image exists
            image_path = self.trial_dir / entry['image_path']
            if not image_path.exists():
                validation['missing_images'] += 1
                validation['issues'].append(f"Missing image at timestep {i}")
            elif validation['image_shape'] is None:
                # Load first image to get shape
                img = cv2.imread(str(image_path))
                if img is not None:
                    validation['image_shape'] = img.shape
            
            # Validate state
            if 'state' in entry:
                for key, value in entry['state'].items():
                    if key not in validation['state_dims']:
                        validation['state_dims'][key] = []
                    validation['state_dims'][key].append(value)
            
            # Check temporal consistency
            if i > 0:
                time_gap = entry['timestamp'] - self.data_log[i-1]['timestamp']
                if time_gap > max_gap:  # More than 1.5x the expected period
                    validation['temporal_gaps'].append({
                        'timestep': i,
                        'gap_ms': time_gap * 1000
                    })
        
        # Print validation results
        print(f"\n✓ Total timesteps: {validation['total_timesteps']}")
        print(f"✓ Image shape: {validation['image_shape']}")
        print(f"✓ Action dimensions: {validation['action_dims']} (vx, vy, vz, yaw_rate)")
        print(f"✓ State variables: {list(validation['state_dims'].keys())}")
        
        if validation['missing_images'] > 0:
            print(f"\n⚠ WARNING: {validation['missing_images']} missing images")
        
        if len(validation['temporal_gaps']) > 0:
            print(f"\n⚠ WARNING: {len(validation['temporal_gaps'])} temporal gaps detected")
            print(f"  (Gaps > {max_gap * 1000:.0f}ms between consecutive timesteps)")
        
        if len(validation['issues']) == 0 and len(validation['temporal_gaps']) == 0:
            print("\n✓ All validation checks passed!")
        
        return validation
    
    def analyze_state_distribution(self):
        """Analyze distribution of state variables."""
        print("\n" + "="*60)
        print("STATE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        state_stats = {}
        
        for key in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw']:
            values = [entry['state'].get(key, 0.0) for entry in self.data_log]
            state_stats[key] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # Print position stats
        print("\nPosition (meters):")
        for axis in ['x', 'y', 'z']:
            stats = state_stats[axis]
            print(f"  {axis}: min={stats['min']:6.3f}, max={stats['max']:6.3f}, "
                  f"mean={stats['mean']:6.3f}, std={stats['std']:6.3f}")
        
        # Print velocity stats
        print("\nVelocity (m/s):")
        for axis in ['vx', 'vy', 'vz']:
            stats = state_stats[axis]
            print(f"  {axis}: min={stats['min']:6.3f}, max={stats['max']:6.3f}, "
                  f"mean={stats['mean']:6.3f}, std={stats['std']:6.3f}")
        
        # Print orientation stats
        print("\nOrientation (degrees):")
        for axis in ['roll', 'pitch', 'yaw']:
            stats = state_stats[axis]
            print(f"  {axis}: min={stats['min']:6.2f}, max={stats['max']:6.2f}, "
                  f"mean={stats['mean']:6.2f}, std={stats['std']:6.2f}")
        
        return state_stats
    
    def analyze_action_distribution(self):
        """Analyze distribution of actions."""
        print("\n" + "="*60)
        print("ACTION DISTRIBUTION ANALYSIS")
        print("="*60)
        
        action_names = ['vx', 'vy', 'vz', 'yaw_rate']
        action_stats = {}
        
        for i, name in enumerate(action_names):
            values = [entry['action'][i] for entry in self.data_log]
            action_stats[name] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'nonzero_ratio': np.sum(np.abs(values) > 0.01) / len(values)
            }
        
        print("\nAction Statistics:")
        for name, stats in action_stats.items():
            print(f"\n  {name}:")
            print(f"    Range: [{stats['min']:6.3f}, {stats['max']:6.3f}]")
            print(f"    Mean: {stats['mean']:6.3f}, Std: {stats['std']:6.3f}")
            print(f"    Non-zero ratio: {stats['nonzero_ratio']*100:.1f}%")
        
        # Check for action diversity
        print("\n" + "-"*60)
        print("Action Diversity Check:")
        
        all_zeros = sum(1 for entry in self.data_log 
                       if all(abs(a) < 0.01 for a in entry['action']))
        if all_zeros > len(self.data_log) * 0.5:
            print(f"  ⚠ WARNING: {all_zeros}/{len(self.data_log)} timesteps have zero actions")
            print("  This suggests the drone was mostly hovering.")
        else:
            print(f"  ✓ Good action diversity: {len(self.data_log) - all_zeros}/{len(self.data_log)} "
                  f"timesteps have non-zero actions")
        
        return action_stats
    
    def visualize_trajectory(self, save_path=None):
        """
        Visualize 3D trajectory of the drone.
        
        Args:
            save_path: Optional path to save the figure
        """
        print("\n" + "="*60)
        print("TRAJECTORY VISUALIZATION")
        print("="*60)
        
        # Extract positions
        x = [entry['state']['x'] for entry in self.data_log]
        y = [entry['state']['y'] for entry in self.data_log]
        z = [entry['state']['z'] for entry in self.data_log]
        timestamps = [entry['timestamp'] for entry in self.data_log]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory
        ax1 = fig.add_subplot(221, projection='3d')
        scatter = ax1.scatter(x, y, z, c=timestamps, cmap='viridis', s=10)
        ax1.plot(x, y, z, 'b-', alpha=0.3, linewidth=0.5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        plt.colorbar(scatter, ax=ax1, label='Time (s)')
        
        # XY projection
        ax2 = fig.add_subplot(222)
        scatter2 = ax2.scatter(x, y, c=timestamps, cmap='viridis', s=10)
        ax2.plot(x, y, 'b-', alpha=0.3, linewidth=0.5)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top-down View (XY)')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        plt.colorbar(scatter2, ax=ax2, label='Time (s)')
        
        # Height over time
        ax3 = fig.add_subplot(223)
        ax3.plot(timestamps, z, 'b-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Height (m)')
        ax3.set_title('Height Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Velocities over time
        ax4 = fig.add_subplot(224)
        vx = [entry['state']['vx'] for entry in self.data_log]
        vy = [entry['state']['vy'] for entry in self.data_log]
        vz = [entry['state']['vz'] for entry in self.data_log]
        ax4.plot(timestamps, vx, label='vx', alpha=0.7)
        ax4.plot(timestamps, vy, label='vy', alpha=0.7)
        ax4.plot(timestamps, vz, label='vz', alpha=0.7)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.set_title('Velocities Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved to: {save_path}")
        else:
            plt.savefig(self.trial_dir / "trajectory_analysis.png", dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved to: {self.trial_dir / 'trajectory_analysis.png'}")
        
        plt.close()
    
    def visualize_actions(self, save_path=None):
        """
        Visualize actions over time.
        
        Args:
            save_path: Optional path to save the figure
        """
        print("\n" + "="*60)
        print("ACTION VISUALIZATION")
        print("="*60)
        
        timestamps = [entry['timestamp'] for entry in self.data_log]
        actions = np.array([entry['action'] for entry in self.data_log])
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        action_names = ['Forward/Backward (vx)', 'Left/Right (vy)', 
                       'Up/Down (vz)', 'Yaw Rate']
        
        for i, (ax, name) in enumerate(zip(axes, action_names)):
            ax.plot(timestamps, actions[:, i], linewidth=1.5)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        axes[-1].set_xlabel('Time (s)')
        axes[0].set_title('Actions Over Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Action plot saved to: {save_path}")
        else:
            plt.savefig(self.trial_dir / "action_analysis.png", dpi=150, bbox_inches='tight')
            print(f"Action plot saved to: {self.trial_dir / 'action_analysis.png'}")
        
        plt.close()
    
    def create_video(self, fps=10, save_path=None):
        """
        Create an MP4 video from all collected images in order.
        
        Args:
            fps: Frames per second for the video (default: 10, matching collection rate)
            save_path: Optional path to save the video (default: trial_dir/trajectory.mp4)
        """
        print("\n" + "="*60)
        print("CREATING VIDEO")
        print("="*60)
        
        if save_path is None:
            save_path = self.trial_dir / "trajectory.mp4"
        
        # Get all image paths in order
        image_files = []
        for entry in self.data_log:
            image_path = self.trial_dir / entry['image_path']
            if image_path.exists():
                image_files.append(image_path)
        
        if len(image_files) == 0:
            print("No images found to create video")
            return
        
        # Read first image to get dimensions
        first_frame = cv2.imread(str(image_files[0]))
        if first_frame is None:
            print("Error: Could not read first image")
            return
        
        height, width, channels = first_frame.shape
        
        # Initialize video writer
        # H.264 ('avc1') so the video plays in QuickTime, VS Code, browsers and slides
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print("Error: Could not open video writer")
            return
        
        # Write all frames
        for i, image_path in enumerate(image_files):
            frame = cv2.imread(str(image_path))
            if frame is not None:
                video_writer.write(frame)
            else:
                print(f"Warning: Could not read frame {i}")
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} frames...")
        
        video_writer.release()
        
        duration = len(image_files) / fps
        print(f"\n✓ Video created: {save_path}")
        print(f"  Frames: {len(image_files)}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Resolution: {width}x{height}")
    
    def create_gif(self, fps=10, save_path=None, width=320, frame_step=2):
        """
        Create a color GIF from all collected images in order.
        
        Args:
            fps: Rate the images were collected at, so the GIF plays at real speed
            save_path: Optional path to save the GIF (default: trial_dir/trajectory.gif)
            width: GIF width in pixels (frames are shrunk to keep the file small)
            frame_step: Keep every frame_step-th image
        """
        print("\n" + "="*60)
        print("CREATING GIF")
        print("="*60)
        
        if save_path is None:
            save_path = self.trial_dir / "trajectory.gif"
        
        frames = []
        for entry in self.data_log[::frame_step]:
            image_path = self.trial_dir / entry['image_path']
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            height = int(frame.shape[0] * width / frame.shape[1])
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        
        if len(frames) == 0:
            print("No images found to create GIF")
            return
        
        # Each GIF frame lasts frame_step collection periods
        frame_ms = int(round(1000 * frame_step / fps))
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
        
        size_mb = save_path.stat().st_size / 1e6
        print(f"\n✓ GIF created: {save_path}")
        print(f"  Frames: {len(frames)} ({frames[0].width}x{frames[0].height}), {size_mb:.1f} MB")
    
    def check_image_samples(self, num_samples=5):
        """
        Display random image samples from the dataset.
        
        Args:
            num_samples: Number of random samples to display
        """
        print("\n" + "="*60)
        print(f"IMAGE SAMPLES ({num_samples} random frames)")
        print("="*60)
        
        # Select random indices
        indices = np.random.choice(len(self.data_log), 
                                  min(num_samples, len(self.data_log)), 
                                  replace=False)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, indices):
            entry = self.data_log[idx]
            image_path = self.trial_dir / entry['image_path']
            
            img = cv2.imread(str(image_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(f"Frame {entry['timestep']}\n"
                           f"t={entry['timestamp']:.2f}s\n"
                           f"z={entry['state']['z']:.2f}m")
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.trial_dir / "image_samples.png", dpi=150, bbox_inches='tight')
        print(f"Image samples saved to: {self.trial_dir / 'image_samples.png'}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60)
        
        report = {
            'trial_name': self.trial_dir.name,
            'analysis_date': datetime.now().isoformat(),
            'metadata': self.metadata,
            'validation': self.validate_data(),
            'state_stats': self.analyze_state_distribution(),
            'action_stats': self.analyze_action_distribution()
        }
        
        # Save report
        report_path = self.trial_dir / "analysis_report.json"
        with open(report_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(report, f, indent=2, default=convert)
        
        print(f"\nSummary report saved to: {report_path}")
        
        return report
    
    def assess_for_il_training(self):
        """
        Assess whether this trial has sufficient data quality for IL training.
        """
        print("\n" + "="*60)
        print("IMITATION LEARNING TRAINING READINESS ASSESSMENT")
        print("="*60)
        
        issues = []
        warnings = []
        
        # Check minimum data size
        if len(self.data_log) < 100:
            issues.append(f"Too few timesteps ({len(self.data_log)} < 100)")
        elif len(self.data_log) < 300:
            warnings.append(f"Limited data ({len(self.data_log)} timesteps). More data recommended.")
        
        # Check action diversity
        all_zeros = sum(1 for entry in self.data_log 
                       if all(abs(a) < 0.01 for a in entry['action']))
        if all_zeros > len(self.data_log) * 0.7:
            issues.append(f"Insufficient action diversity ({all_zeros}/{len(self.data_log)} zero actions)")
        
        # Check state variation
        z_values = [entry['state']['z'] for entry in self.data_log]
        if np.std(z_values) < 0.05:
            warnings.append("Very limited height variation - consider more diverse maneuvers")
        
        # Print assessment
        if len(issues) == 0:
            print("\n✓ READY FOR IL TRAINING")
            if len(warnings) > 0:
                print("\nWarnings:")
                for w in warnings:
                    print(f"  ⚠ {w}")
        else:
            print("\n✗ NOT READY - Issues found:")
            for issue in issues:
                print(f"  ✗ {issue}")
            if len(warnings) > 0:
                print("\nAdditional warnings:")
                for w in warnings:
                    print(f"  ⚠ {w}")
        
        return len(issues) == 0


def analyze_trial(trial_path, generate_plots=True):
    """
    Main function to analyze a single trial.
    
    Args:
        trial_path: Path to trial directory
        generate_plots: Whether to generate visualization plots
    """
    processor = ImitationDataProcessor(trial_path)
    
    # Validate data
    processor.validate_data()
    
    # Analyze distributions
    processor.analyze_state_distribution()
    processor.analyze_action_distribution()
    
    # Generate visualizations
    if generate_plots:
        processor.visualize_trajectory()
        processor.visualize_actions()
        processor.check_image_samples()
        
        # Create video + GIF from images at the rate actually achieved while recording
        # (can be a little under the target rate), so both play back at real speed
        if processor.metadata:
            rate_hz = processor.metadata['total_timesteps'] / processor.metadata['duration']
        else:
            rate_hz = 10
        processor.create_video(fps=rate_hz)
        processor.create_gif(fps=rate_hz)
    
    # Generate summary report
    processor.generate_summary_report()
    
    # Assess training readiness
    processor.assess_for_il_training()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


def analyze_all_trials(data_dir=Path(__file__).resolve().parent / 'imitation_data'):
    """
    Analyze all trials in the data directory.
    
    Args:
        data_dir: Path to imitation data directory
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Find all trial directories
    trials = sorted([d for d in data_path.iterdir() 
                    if d.is_dir() and d.name.startswith('trial_')])
    
    if len(trials) == 0:
        print(f"No trials found in {data_dir}")
        return
    
    print("="*60)
    print(f"ANALYZING ALL TRIALS ({len(trials)} found)")
    print("="*60)
    
    for trial_dir in trials:
        print(f"\n{'='*60}")
        print(f"Processing: {trial_dir.name}")
        print(f"{'='*60}")
        
        try:
            analyze_trial(trial_dir, generate_plots=True)
        except Exception as e:
            print(f"Error processing {trial_dir.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL TRIALS ANALYZED")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Analyze specific trial
        trial_path = sys.argv[1]
        analyze_trial(trial_path)
    else:
        # Analyze all trials
        analyze_all_trials()