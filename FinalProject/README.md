# Final Project: Vision-Based Obstacle Avoidance with Imitation Learning

In the final project, we will train a policy that flies the Crazyflie through an obstacle course using only its camera. The drone will take off from the start position and fly toward the goal region while avoiding obstacles. We will do this with behavior cloning, as discussed in class (Lecture 20).

Main steps:

1. **Camera:** connect the receiver and check the picture (`camera_test.ipynb`).
2. **Collect demos:** fly the course with the keyboard while recording (`keyboard_control_IL.py`).
3. **Check each demo:** done automatically when you land (plots + a video of what the camera saw).
4. **Build your dataset:** the TAs have already collected 38 demos; you can collect some more.
5. **Train a model:** your own design, fed by `dataloader.py`.
6. **Fly it:** run your model in a control loop. **The camera image must be processed exactly as in training.**
7. **Present on demo day:** 4-5 slides on your technical approach.

Everything is in the final project folder of the course repo, `F2026/FinalProject/`. Run every command below from there (`cd F2026/FinalProject` first).

| File | What it is |
| --- | --- |
| `camera_test.ipynb` | Connect the receiver, live view, crop check (plus optional record/flight tests) |
| `keyboard_control_IL.py` | Fly with the keyboard and record synchronized (image, state, action) at 20 Hz. Its control loop provides an example of reading the camera while flying |
| `analyze_il_data.py` | Checks a demo and makes plots + a video of the camera images. Runs automatically after each demo |
| `dataloader.py` | PyTorch `Dataset`/`DataLoader` over a folder of demos, plus `preprocess_frame` for flight time |
| `il_dataset.zip` | 38 demos recorded by the course staff (1.2 GB). Not in the repo: [download from Google Drive](https://drive.google.com/file/d/1hgEG0j39_dQzKIepr9XRy3im8qH9uI3t/view?usp=drive_link) |

## 0. Setup

Data collection and the notebook run in the `mae345` env, with two extra packages:

```
conda activate mae345
pip install cflib pynput
```

Training needs PyTorch, which conflicts with the `mae345` env, so make a second env. It also needs cflib and pynput, so you can fly your trained model from it:

```
conda create -n mae345-il python=3.11
conda activate mae345-il
pip install torch torchvision opencv-python numpy matplotlib cflib pynput
```

**macOS permissions** (one time):
- **Camera:** *System Settings → Privacy & Security → Camera* → allow the app you run Python from (Terminal / iTerm / VS Code).
- **Keyboard:** the data collection script reads keys globally, so that same app needs *Privacy & Security → Input Monitoring* (and *Accessibility* if keys still aren't seen). Restart the app after granting.

## 1. Camera: `camera_test.ipynb`

```
conda activate mae345
jupyter lab
```

Run **Parts 0–3**:

- **Part 0:** mount the camera on the drone and plug the receiver into your Mac.
- **Part 1:** find the receiver's camera index (the 640x480 one). Note this number.
- **Part 2:** live view. Power on the drone, then hold the receiver's left button until it locks onto your camera. Wave a hand in front of the camera to check the picture is yours.
- **Part 3:** check that the crop removes the receiver's overlay text.

Parts 4–6 are optional checks: record a clip, a hover test, fly while recording. They're useful if something misbehaves later.

Then **shut down the kernel** (Kernel → Shut Down) so it releases the camera and radio.

## 2. Collect demos: `keyboard_control_IL.py`

Every demo is a complete run: take off from the nominal start position and fly to the goal region (the end box) while avoiding the obstacles. How many demos you collect is up to you; more, consistent demos generally help. Rearrange the poles between demos so they cover several layouts. Discard bad runs with `n`.

From a terminal (the keyboard listener doesn't work inside Jupyter):

```
conda activate mae345
python keyboard_control_IL.py --group <your drone's radio channel> --camera <index from Part 1>
```

- **Trial number:** you're asked for one (number them 1, 2, 3, …). Existing trials are listed, and you're asked before one is overwritten.
- **Setup wait:** wait for "Connected!" and "Estimator converged", with the drone flat and still.
- **Recording:** press `u`. The drone takes off, hovers 3 s, and **recording starts automatically**. Fly the course, then press `l` to save and land.
- **Where it goes:** demos are saved to `F2026/FinalProject/imitation_data/trial_XX/`. Use `--out <folder>` to save somewhere else.

**Keys** (velocity 0.2 m/s, yaw rate 45°/s):

| Key | Action |
| --- | --- |
| `u` | arm + take off (0.5 m), 3 s hover, then recording starts |
| `w` / `s` | forward / back |
| `a` / `d` | left / right |
| `w`+`a`, `w`+`d` | **diagonal**: forward-left / forward-right. Hold both keys at once |
| `space` / `c` | up / down |
| `q` / `e` | turn left / right |
| `l` | stop recording, **save**, land |
| `n` | stop recording, **discard** the demo, land |
| `esc` | **emergency stop**: cuts the motors, the drone drops. Data is saved |

Keys go to the script even when the terminal isn't focused, so don't type into other windows while flying.

### Tips for good demos

The model can only be as consistent as your demos. These are some things that the TAs learned from collecting data:

- **Pick one way of avoiding obstacles and stick to it.** Holding `w`+`a` / `w`+`d` to curve around an obstacle while still moving forward works well. If you sometimes curve and sometimes stop and slide sideways, the same image gets two different labels.
- **Stay at the takeoff height and keep facing forward.** Climbing (`space`/`c`), turning (`q`/`e`) and backing up (`s`) produce camera views and commands your policy will rarely see. Avoid them unless you plan to use them on purpose.
- **Don't stop and hover mid-course.** Every frame you hover teaches the policy to hover in front of obstacles.
- **Start from the nominal start position every time, and finish in the goal region.** The poles can and should move between demos (the policy has to handle new layouts), but the start and the goal region don't.
- **Discard bad runs with `n`.** A crash or a wrong turn is better thrown away than trained on.

### Cropping

The receiver draws an overlay along the top of the image: signal strength, frequency, signal strength (e.g. `92%  5789MHz  94%`). The script cuts off the top 80 rows of each frame (`--crop`, default 80) so the overlay doesn't end up in your data. The receiver doesn't always show the overlay, but when it does, the text can reach down to about row 105. If you see text in notebook Part 3 or in a demo's `image_samples.png`, record with `--crop 110`. Either is fine: the data loader lines up demos recorded at different crops (step 4).

## 3. Check each demo

When you land, the script runs `analyze_il_data.py` on the demo (skip it with `--no-analysis`). It writes these into the trial folder:

| File | What to look for |
| --- | --- |
| `trajectory.mp4` | Video of the camera images, at real speed: exactly what your policy will see. Open it with `open imitation_data/trial_XX/trajectory.mp4` |
| `image_samples.png` | No overlay text at the top, no static |
| `trajectory_analysis.png` | The flight path: did it go where you meant? |
| `action_analysis.png` | The commands you sent |

It also prints the collection rate (about 18 Hz is normal) and any gaps in the data. To re-run it: `python analyze_il_data.py imitation_data/trial_XX`, or `python analyze_il_data.py` for every trial. Discard or re-fly any demo that looks wrong.

## 4. Build your dataset

Each demo is a folder:

```
trial_XX/
  images/frame_000000.jpg ...   camera frames, top crop_top rows already removed
  data_log.json                 one entry per frame: image_path, state, action, timestamp
  metadata.json                 crop_top, image_shape, collection rate, ...
```

- **state:** `x, y, z, vx, vy, vz, roll, pitch, yaw`, from the Flow deck and the drone's estimator.
- **action:** the command you were sending, `[vx, vy, vz, yaw_rate]` in m/s and deg/s.

1. **Download the staff demos:** [`il_dataset.zip`](https://drive.google.com/file/d/1hgEG0j39_dQzKIepr9XRy3im8qH9uI3t/view?usp=drive_link) (1.2 GB, Google Drive, Princeton login), then move it into `F2026/FinalProject/` and unzip it there:

   ```
   mv ~/Downloads/il_dataset.zip .
   unzip il_dataset.zip
   ```

   This gives `F2026/FinalProject/il_dataset/trial_01` … `trial_38`. If your browser already unzipped it (Safari does), move the `il_dataset` folder from `~/Downloads` instead. The staff demos are recorded at crop 110 and come from several pilots, steering styles and pole layouts. Watch a few of their videos: `python analyze_il_data.py il_dataset/trial_01`, then open `il_dataset/trial_01/trajectory.mp4`.
2. **Copy your demos in** after the staff ones: your `trial_01` becomes `trial_39`, your `trial_02` becomes `trial_40`, and so on. The data loader reads one folder. This copies every demo in `imitation_data/` (your originals stay where they are):

   ```
   i=39; for d in imitation_data/trial_*; do cp -r "$d" "il_dataset/trial_$i"; i=$((i+1)); done
   ```

   When you're done, the folder looks like this:

   ```
   F2026/FinalProject/
     dataloader.py, keyboard_control_IL.py, ...
     imitation_data/   trial_01, trial_02, ...         your demos, as recorded
     il_dataset/       trial_01 ... trial_38 (staff)   what you train on
                       trial_39, trial_40, ... (yours)
   ```

3. **Check the result:** `python dataloader.py` (in `mae345-il`) loads everything and prints the action set and how often each action occurs. It also checks that flight-time preprocessing matches training (step 6).

**Mixed crops:** every image has to cover the same part of the camera frame, so the loader crops every demo to the **largest** crop in the folder. With the staff demos that's 110, so your crop-80 demos lose 30 more rows when loaded. Nothing changes on disk. Removed rows can't be put back, so you can't train at a smaller crop than any demo was recorded with.

## 5. Train a model

The model and training loop are yours to design. The data loader gives you batches:

```python
from dataloader import create_dataloaders

train_loader, val_loader = create_dataloaders('il_dataset', batch_size=32)
batch = next(iter(train_loader))
batch['observation']   # [B, 3, H, W] images, cropped, resized to H x W = 144 x 256, ImageNet-normalized
batch['state']         # [B, 9]   standardized (see normalize_states)
batch['action']        # [B, 4]   vx, vy, vz, yaw_rate, as commanded
batch['action_class']  # [B]      index of the action in the action set
```

What the loader does for you:

- **Crop:** brings every demo to the same crop (step 4).
- **Idle frames:** drops the hover before your first command and after your last.
- **Split:** splits whole demos into train/validation, seeded. Splitting individual frames would leak, since neighbouring frames are nearly identical.
- **Augmentation (training only):** small shifts and color jitter. It also mirrors images left-right and **mirrors the labels with them** (left becomes right). Flipping only the image would teach the wrong direction.
- **Validation:** uses the training set's normalization and action set.

**Discrete actions:** the keyboard only produces a handful of distinct commands (forward, forward-left, slide right, ...), listed in `train_loader.dataset.action_set`. You can treat the problem as classification over those (`action_class`), or as regression on `action`. Which works better is a design choice worth exploring. 

**When you save your model, save the data settings with it:**

```python
torch.save({'model': model.state_dict(),
            'stats': train_loader.dataset.get_stats()},   # crop, image size, normalization, action set
           'my_policy.pt')
```

## 6. Fly your policy: inference must match training

At flight time, your model has to receive exactly the kind of input it was trained on: same crop, same color order (RGB), same resize, same normalization. `preprocess_frame` does all of that from the raw 640x480 camera frame, using the settings you saved:

```python
import torch
from dataloader import preprocess_frame

ckpt = torch.load('my_policy.pt')
model = MyModel(...)
model.load_state_dict(ckpt['model'])
model.eval()
stats = ckpt['stats']

# In your control loop, with cap = cv2.VideoCapture(<receiver index>) and mc = MotionCommander(scf):
ret, frame = cap.read()                       # the raw frame. Don't crop it yourself
x = preprocess_frame(frame, stats['crop_top'], tuple(stats['image_size']))   # [1, 3, H, W]
with torch.no_grad():
    out = model(x)
vx, vy, vz, yaw_rate = ...                    # e.g. stats['action_set'][out.argmax()] for a classifier
mc.start_linear_motion(vx, vy, vz, yaw_rate)
```

- **Normalized targets:** if you trained on normalized actions (`normalize_actions=True`), undo it before sending: `action * stats['action_stats']['std'] + stats['action_stats']['mean']`. If your model also takes the state, normalize it with `stats['state_stats']` the same way.
- **Any processing you add in training** (a different crop, image size, grayscale, ...) has to happen at flight time too. Put it in one function and call it from both places.
- **The control loop:** `keyboard_control_IL.py` shows how to connect, arm (`send_arming_request`), set up the PID controller, take off, send velocity commands at 20 Hz and land. Your flight script can follow the same structure, with the model in place of the keyboard.
- **Landing:** your demos stop recording when you press `l`, so a policy trained on them never sees a landing and can't learn when to land. Decide that in your flight script. One option: land once the drone's position estimate (`stateEstimate.x`, logged as in `keyboard_control_IL.py`) shows it has flown as far forward as your demos did.
- **Safety:** always keep a way to land at once (e.g. Ctrl-C → `mc.land()`) and a time limit on the loop.

## 7. Demo day: flights

See slides 29-30 of Lecture 20 for details on how your flights will be scored. As mentioned there, you will have two flights under nominal conditions, 1 flight under a lighting change, and 1 flight with different colored obstacles.

## 8. Demo day: 4–5 slides

Along with flying your policy on demo day, present **4–5 slides** on your technical approach. For example:

- **Model:** neural network architecture, inference latency, etc.
- **Data:** how many additional demos you collected, how you flew them, and anything you filtered, cleaned, or weighted.
- **Training:** loss, data augmentation, how long you trained, and how you picked the final checkpoint.
- **What you tried:** the alternatives you compared, what didn't work, and how the final model does in flight.

We hope you enjoy the project! 
