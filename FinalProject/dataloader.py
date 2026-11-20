"""
PyTorch Dataset / DataLoader for the Crazyflie imitation learning demos.

Data layout (one folder per demo, written by keyboard_control_IL.py):
    il_dataset/trial_XX/{images/frame_XXXXXX.jpg, data_log.json, metadata.json}

Each sample is a dict:
    observation   [3, H, W] image, normalized with ImageNet mean/std
    state         [9]  x, y, z, vx, vy, vz, roll, pitch, yaw
    action        [4]  commanded vx, vy, vz, yaw_rate (m/s, m/s, m/s, deg/s)
    action_class  []   index of `action` in the dataset's action set (for
                       classification policies; see ACTION_SET notes below)

Things this loader takes care of, so every consumer gets the same thing:
  * Crop. Every image must cover the same rows of the receiver's 640x480
    frame, in training AND in flight. Demos are recorded at crop 80 by default,
    but some may have needed more (e.g. the staff demos in il_dataset/ are all
    at 110). By default (`crop_top=None`) the loader uses the LARGEST crop
    among the trials and crops the rest down to it on load. The crop used is
    saved with the model so flying applies the same one (preprocess_frame).
  * Idle frames. The hover before the pilot starts moving and after they stop
    is labelled "do nothing"; a policy trained on it learns to hover at the
    start line. `trim_idle=True` (default) drops those leading/trailing frames.
  * Mirror augmentation. A left-right flipped image is only a valid sample if
    the labels flip too (vy, yaw_rate; y, vy, roll, yaw in the state).
  * Normalization stats and the action set come from the training split and
    are reused for validation. `get_stats()` returns them (crop included):
    save them with your model, since flying needs the same ones.

When flying, run the live 640x480 frame through
`preprocess_frame(frame, crop_top, image_size)` with the crop_top and
image_size you trained with, so the network sees exactly what it was trained on.
"""

import json
import random
import numpy as np
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'il_dataset'
IMAGE_SIZE = (144, 256)        # (H, W) fed to the network; keeps the 370x640 aspect ratio
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STATE_KEYS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw']
ACTION_KEYS = ['vx', 'vy', 'vz', 'yaw_rate']

# Mirroring the image left-right negates lateral and rotational quantities.
MIRROR_STATE = np.array([1, -1, 1, 1, -1, 1, -1, 1, -1], dtype=np.float32)  # y, vy, roll, yaw
MIRROR_ACTION = np.array([1, -1, 1, -1], dtype=np.float32)                  # vy, yaw_rate


def _image_transform(image_size, augment=False):
    tf = [transforms.ToPILImage(), transforms.Resize(image_size)]
    if augment:
        # Small shifts / zoom / roll: the camera's mounting angle differs between
        # sessions (measured: ~10-30 px of tilt and a little roll), and this also
        # fights overfitting to the exact framing of the training trials.
        # Symmetric, so it doesn't interfere with the left-right mirroring.
        tf.append(transforms.RandomAffine(degrees=3, translate=(0.05, 0.08), scale=(0.95, 1.08)))
        tf.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    tf += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(tf)


def preprocess_frame(frame_bgr: np.ndarray, crop_top: int,
                     image_size: Tuple[int, int] = IMAGE_SIZE) -> torch.Tensor:
    """
    Turn a raw 640x480 BGR frame from the receiver into the network's input,
    exactly as the dataset does (crop, BGR->RGB, resize, normalize).
    `crop_top` and `image_size` must be the ones the model was trained with
    (dataset.crop_top, dataset.image_size). Returns a [1, 3, H, W] tensor.
    """
    if frame_bgr.shape[0] != 480:
        raise ValueError(f"expected the receiver's raw 480-row frame, got {frame_bgr.shape}; "
                         f"pass the frame before any cropping")
    image = cv2.cvtColor(frame_bgr[crop_top:], cv2.COLOR_BGR2RGB)
    return _image_transform(image_size)(image).unsqueeze(0)


def mean_forward_distance(data_dir=DEFAULT_DATA_DIR, trial_numbers=None) -> float:
    """Average distance (m) the demos flew forward, from the first logged frame to
    the last (where the pilot pressed 'l' to land)."""
    dists = []
    for d in sorted(Path(data_dir).iterdir()):
        if d.is_dir() and d.name.startswith('trial_') and (d / 'data_log.json').exists():
            if trial_numbers is None or int(d.name.split('_')[1]) in trial_numbers:
                log = json.load(open(d / 'data_log.json'))
                dists.append(log[-1]['state']['x'] - log[0]['state']['x'])
    if not dists:
        raise ValueError(f"No trials found in {data_dir} (trial_numbers={trial_numbers})")
    return float(np.mean(dists))


def recorded_crops(data_dir=DEFAULT_DATA_DIR, trial_numbers=None) -> Dict[int, int]:
    """{trial number: crop_top it was recorded with} from each metadata.json."""
    crops = {}
    for d in sorted(Path(data_dir).iterdir()):
        if d.is_dir() and d.name.startswith('trial_') and (d / 'metadata.json').exists():
            n = int(d.name.split('_')[1])
            if trial_numbers is None or n in trial_numbers:
                crops[n] = json.load(open(d / 'metadata.json')).get('crop_top')
    missing = [n for n, c in crops.items() if c is None]
    if missing:
        raise ValueError(f"trials {missing} have no crop_top in metadata.json, can't check the crop")
    return crops


def _action_key(action) -> Tuple[float, ...]:
    return tuple(round(float(a), 3) for a in action)


def build_action_set(actions) -> List[Tuple[float, ...]]:
    """Sorted list of the distinct actions, closed under mirroring (so a
    flipped sample's label is always in the set)."""
    keys = {_action_key(a) for a in actions}
    keys |= {_action_key(np.array(k) * MIRROR_ACTION) for k in keys}
    return sorted(keys)


class CrazyflieILDataset(Dataset):
    """PyTorch Dataset over the trial folders in `data_dir`."""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        trial_numbers: Optional[List[int]] = None,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        normalize_states: bool = True,
        normalize_actions: bool = False,
        augment: bool = False,
        trim_idle: bool = True,
        crop_top: Optional[int] = None,
        stats: Optional[Dict] = None,
        action_set: Optional[List[List[float]]] = None,
        verbose: bool = True,
    ):
        """
        Args:
            data_dir: Root directory containing trial_XX folders
            trial_numbers: Trial numbers to include (None = all trials)
            image_size: (H, W) the images are resized to
            normalize_states: Standardize the state vector
            normalize_actions: Standardize the continuous action vector
            augment: Color jitter + label-consistent mirroring (training only)
            trim_idle: Drop the zero-action frames before the first and after
                       the last command of each trial
            crop_top: Rows removed from the top of the raw 480-row frame.
                   None = the largest crop any of these trials was recorded with.
            stats: `get_stats()` of the training set. Pass it to validation/test
                   sets so they use the training crop, normalization and action
                   set. None = compute from this dataset.
            action_set: The actions a classifier chooses between. None = the
                   distinct actions in this dataset (plus their mirrors).
                   create_dataloaders passes the set from ALL trials, so an
                   action that only appears in validation still has a class.
        """
        self.data_dir = Path(data_dir)
        self.image_size = tuple(image_size)
        self.normalize_states = normalize_states
        self.normalize_actions = normalize_actions
        self.augment = augment
        self.trim_idle = trim_idle
        if stats is not None:
            if crop_top is not None and crop_top != stats['crop_top']:
                raise ValueError(f"crop_top={crop_top} differs from the training crop {stats['crop_top']}")
            crop_top = stats['crop_top']
        if crop_top is None:
            crop_top = max(recorded_crops(data_dir, trial_numbers).values())
        self.crop_top = crop_top

        self.trials = []
        self.samples = []
        self._load_trials(trial_numbers)

        if stats is None:
            stats = self._compute_stats()
            if action_set is not None:
                stats['action_set'] = [list(a) for a in action_set]
        self.state_stats = stats['state_stats']
        self.action_stats = stats['action_stats']
        self.action_set = [tuple(a) for a in stats['action_set']]
        self._action_index = {a: i for i, a in enumerate(self.action_set)}
        for t, i in self.samples:
            a = self.trials[t]['data_log'][i]['action']
            if _action_key(a) not in self._action_index:
                raise ValueError(f"trial {self.trials[t]['trial_number']}: action {a} is not in the "
                                 f"training action set {self.action_set}")

        self.transform = _image_transform(self.image_size, augment)

        if verbose:
            print(f"Dataset: {len(self.trials)} trials {[t['trial_number'] for t in self.trials]}, "
                  f"{len(self.samples)} samples ({self.n_trimmed} idle frames trimmed), "
                  f"images {self.image_size} after a {crop_top}-row crop, augment={augment}")

    # ------------------------------------------------------------------ loading
    def _load_trials(self, trial_numbers: Optional[List[int]]):
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        trial_dirs = sorted(d for d in self.data_dir.iterdir()
                            if d.is_dir() and d.name.startswith('trial_'))
        if trial_numbers is not None:
            trial_dirs = [d for d in trial_dirs if int(d.name.split('_')[1]) in trial_numbers]
        if not trial_dirs:
            raise ValueError(f"No trials found in {self.data_dir} (trial_numbers={trial_numbers})")

        self.n_trimmed = 0
        for trial_dir in trial_dirs:
            trial = self._load_single_trial(trial_dir)
            if trial is None:
                continue
            self.trials.append(trial)
            for i in range(trial['start'], trial['end']):
                self.samples.append((len(self.trials) - 1, i))

    def _load_single_trial(self, trial_dir: Path) -> Optional[Dict]:
        data_log_path = trial_dir / 'data_log.json'
        metadata_path = trial_dir / 'metadata.json'
        if not data_log_path.exists() or not metadata_path.exists():
            print(f"Warning: skipping {trial_dir.name} - data_log.json or metadata.json missing")
            return None
        data_log = json.load(open(data_log_path))
        metadata = json.load(open(metadata_path))

        recorded_crop = metadata.get('crop_top')
        if recorded_crop is None:
            raise ValueError(f"{trial_dir.name}: metadata.json has no crop_top, can't check the crop")
        if recorded_crop > self.crop_top:
            raise ValueError(f"{trial_dir.name} was recorded with crop {recorded_crop}, more than "
                             f"crop_top={self.crop_top}; those rows are gone. Use crop_top>={recorded_crop}.")

        start, end = 0, len(data_log)
        if self.trim_idle:
            moving = [i for i, e in enumerate(data_log) if any(a != 0 for a in e['action'])]
            if not moving:
                print(f"Warning: skipping {trial_dir.name} - no non-zero actions")
                return None
            start, end = moving[0], moving[-1] + 1
            self.n_trimmed += len(data_log) - (end - start)

        return {
            'trial_dir': trial_dir,
            'trial_number': int(trial_dir.name.split('_')[1]),
            'data_log': data_log,
            'metadata': metadata,
            'extra_crop': self.crop_top - recorded_crop,
            'start': start,
            'end': end,
        }

    def _compute_stats(self) -> Dict:
        entries = [self.trials[t]['data_log'][i] for t, i in self.samples]
        states = np.array([[e['state'][k] for k in STATE_KEYS] for e in entries], dtype=np.float32)
        actions = np.array([e['action'] for e in entries], dtype=np.float32)
        if self.augment:  # mirrored samples are part of the training distribution
            states = np.concatenate([states, states * MIRROR_STATE])
            actions = np.concatenate([actions, actions * MIRROR_ACTION])
        return {
            'state_stats': {'mean': torch.from_numpy(states.mean(0)), 'std': torch.from_numpy(states.std(0) + 1e-6)},
            'action_stats': {'mean': torch.from_numpy(actions.mean(0)), 'std': torch.from_numpy(actions.std(0) + 1e-6)},
            'action_set': build_action_set(actions),
            'crop_top': self.crop_top,
        }

    # ------------------------------------------------------------------ access
    def get_stats(self) -> Dict:
        """Everything about the data a trained model depends on: crop, image size,
        normalization stats and action set. Pass it as `stats=` to a
        validation/test dataset, and save it with your model: flying needs
        preprocess_frame(frame, stats['crop_top'], stats['image_size'])."""
        return {'state_stats': self.state_stats, 'action_stats': self.action_stats,
                'action_set': [list(a) for a in self.action_set], 'crop_top': self.crop_top,
                'image_size': list(self.image_size)}

    def all_actions(self) -> np.ndarray:
        """(N, 4) array of every sample's action."""
        return np.array([self.trials[t]['data_log'][i]['action'] for t, i in self.samples], dtype=np.float32)

    def action_counts(self) -> Dict[Tuple[float, ...], int]:
        counts = {a: 0 for a in self.action_set}
        for t, i in self.samples:
            counts[_action_key(self.trials[t]['data_log'][i]['action'])] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t, i = self.samples[idx]
        trial = self.trials[t]
        entry = trial['data_log'][i]

        image_path = trial['trial_dir'] / entry['image_path']
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image[trial['extra_crop']:], cv2.COLOR_BGR2RGB)

        state = np.array([entry['state'][k] for k in STATE_KEYS], dtype=np.float32)
        action = np.array(entry['action'], dtype=np.float32)

        # Mirror augmentation: flip the image AND the left/right parts of the labels
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            state = state * MIRROR_STATE
            action = action * MIRROR_ACTION

        action_class = self._action_index[_action_key(action)]
        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        if self.normalize_states:
            state = (state - self.state_stats['mean']) / self.state_stats['std']
        if self.normalize_actions:
            action = (action - self.action_stats['mean']) / self.action_stats['std']

        return {
            'observation': self.transform(image),
            'state': state,
            'action': action,
            'action_class': torch.tensor(action_class, dtype=torch.long),
        }


def split_trials(data_dir=DEFAULT_DATA_DIR, val_fraction: float = 0.2, seed: int = 0):
    """Random train/val split over whole trials (frames within a trial are
    near-duplicates, so splitting frames would leak). Seeded = same every run."""
    trials = sorted(int(d.name.split('_')[1]) for d in Path(data_dir).iterdir()
                    if d.is_dir() and d.name.startswith('trial_'))
    random.Random(seed).shuffle(trials)
    n_val = max(1, round(len(trials) * val_fraction))
    return sorted(trials[n_val:]), sorted(trials[:n_val])


def create_dataloaders(
    data_dir: str = DEFAULT_DATA_DIR,
    train_trials: Optional[List[int]] = None,
    val_trials: Optional[List[int]] = None,
    batch_size: int = 32,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    normalize_states: bool = True,
    normalize_actions: bool = False,
    num_workers: int = 4,
    shuffle_train: bool = True,
    val_fraction: float = 0.2,
    seed: int = 0,
    trim_idle: bool = True,
    crop_top: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Train (augmented) and validation dataloaders. With no trial lists given,
    trials are split at random (seeded) with `val_fraction` held out.
    crop_top=None uses the largest crop any trial (train or val) was recorded
    with, so both splits see the same rows. Validation uses the training set's
    crop, normalization stats and action set.
    """
    if crop_top is None:
        crops = recorded_crops(data_dir, None if train_trials is None and val_trials is None
                               else list(train_trials or []) + list(val_trials or []))
        crop_top = max(crops.values())
        print(f"Recorded crops: {sorted(set(crops.values()))} -> training at crop_top={crop_top}")
    if train_trials is None and val_trials is None:
        train_trials, val_trials = split_trials(data_dir, val_fraction, seed)
        print(f"Auto-split: train trials {train_trials}, val trials {val_trials}")

    common = dict(data_dir=data_dir, image_size=image_size, normalize_states=normalize_states,
                  normalize_actions=normalize_actions, trim_idle=trim_idle, crop_top=crop_top)
    # Action set from every trial in both splits, so validation can't contain
    # an action the classifier has no output for.
    all_actions = [CrazyflieILDataset(trial_numbers=trials, verbose=False, **common).all_actions()
                   for trials in (train_trials, val_trials)]
    action_set = build_action_set(np.concatenate(all_actions))
    train_dataset = CrazyflieILDataset(trial_numbers=train_trials, augment=True,
                                       action_set=action_set, **common)
    val_dataset = CrazyflieILDataset(trial_numbers=val_trials, augment=False,
                                     stats=train_dataset.get_stats(), **common)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=torch.cuda.is_available(),  # only useful (and warning-free) on CUDA
                     persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_dataset, shuffle=shuffle_train, **loader_kw)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kw)
    return train_loader, val_loader


if __name__ == '__main__':
    # Quick check: loads every trial, one sample, one batch from each loader.
    dataset = CrazyflieILDataset()
    print("Action set and counts:")
    for a, n in dataset.action_counts().items():
        print(f"  {a}: {n}")
    sample = dataset[0]
    print({k: tuple(v.shape) for k, v in sample.items()})

    train_loader, val_loader = create_dataloaders(batch_size=8, num_workers=0)
    for name, loader in [('train', train_loader), ('val', val_loader)]:
        batch = next(iter(loader))
        print(name, {k: tuple(v.shape) for k, v in batch.items()})

    # Flight-time check: rebuild the raw 480x640 camera frame for a few samples
    # (the recorded crop's rows put back, as blank), run it through
    # preprocess_frame with the saved stats, and compare to the training input.
    val_ds = val_loader.dataset
    stats = train_loader.dataset.get_stats()
    for k in range(0, len(val_ds), max(1, len(val_ds) // 5)):
        t, i = val_ds.samples[k]
        trial = val_ds.trials[t]
        saved = cv2.imread(str(trial['trial_dir'] / trial['data_log'][i]['image_path']))
        raw = np.vstack([np.zeros((trial['metadata']['crop_top'], 640, 3), np.uint8), saved])
        live = preprocess_frame(raw, stats['crop_top'], tuple(stats['image_size']))[0]
        if not torch.equal(live, val_ds[k]['observation']):
            raise AssertionError(f"preprocess_frame doesn't match the dataset for {trial['trial_dir'].name}")
    print(f"✓ preprocess_frame(raw frame) matches the training input (crop {stats['crop_top']}, "
          f"image size {tuple(stats['image_size'])})")
    print("\n✓ DataLoader test passed!")
