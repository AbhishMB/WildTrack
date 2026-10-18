# WildTrack: an autonomous target tracking drone

A pilot flies a drone to where an animal was last seen. A laptop runs **YOLO** on the live video, and the
operator locks onto one animal. The laptop then takes over the flight, follows the animal at a safe distance,
estimates its position on the ground and reports it, with an error radius, on a live dashboard.

> **Academic project.** Built by Abhishek Manohar Bilagi as a **minor project for a B.E./B.Tech in Computer Science
> and Engineering (CSE)**. Its aim is to **prove the concept**. Because flying wildlife drones needs permits and
> hardware, a **full simulation** of the drone, the camera and the animal is included, so the whole idea can be seen
> working on any laptop with no drone at all.

**History and status:** this is a clean re-implementation of my B.E. minor project (2025), which flew a DJI Tello
at low altitude as a proof of concept, with a YOLO model trained on public aerial wildlife datasets. The control logic,
geolocation, safety rules, dashboard and simulator are covered by 148 automated tests, and the Tello link is tested
against a software stand-in of the drone. Trained detector weights are not shipped: [`training/`](training/README.md)
rebuilds them from public aerial datasets in three commands, and until you run it the app uses COCO weights as a
stand-in. As with any new setup, the drone-specific values (speed scale, camera angle, video lag) are set by a short
[calibration](#2-run-it-with-a-real-drone) on your own drone.

---

## How it works

```
 drone camera --video--> laptop --YOLO + tracker--> boxes --pixel to ground--> animal position (+ error radius)
                            ^                                                       |
                            |                                              Kalman filter (position, velocity)
   drone <--velocity cmds-- laptop <------ PID follow controller <-- standoff point beside the animal
```

| Mode | Who flies | What runs |
| --- | --- | --- |
| **MANUAL** | Operator (keyboard) | Detection and boxes; candidate animals are highlighted |
| **LOCKED** | Operator | One animal held by track ID; its position starts publishing; ENGAGE becomes available |
| **AUTO** | The laptop | Follow controller, position updates every second |
| **LOST** | The laptop (hover) | Animal not seen for 2 s: hover; after 10 s: hand control back |

Key design points:
- **Lock, then engage:** two separate operator steps, so a false detection never makes the drone chase something.
- **Only target species can be locked**; a **person can never be locked**, enforced in code (`lock.py`), not just in the UI.
- **Follows a ground position, not pixels:** it estimates where the animal is and flies to a standoff point beside it,
  so a short occlusion (trees, bushes) doesn't lose the animal.
- **PID controller with velocity feedforward** (`controller.py`), with speed, acceleration and integrator limits. The
  derivative is taken from the animal's and the drone's velocities rather than by differencing the error, so a jump
  in the setpoint gives no derivative kick.
- **Safety:** RELEASE / `Esc` always works; the laptop never re-enables autopilot by itself after the pilot (or the
  drone) takes over; battery, range and link-loss checks hand control back.
- **Everything is configurable** in one file, `laptop/config.yaml`; there are no magic numbers in the code.

## Repository layout

```
wildtrack/
  laptop/                    Python 3.11: the whole application
    wildtrack/
      config.py              loads config.yaml (strict: typos are errors)
      detect.py              YOLO + ByteTrack
      lock.py                candidate / lock / re-acquire rules
      geo.py                 pixel -> ground position, error radius
      estimator.py           constant-velocity Kalman filter
      controller.py          standoff setpoint, PID + feedforward, limits
      fsm.py                 MANUAL / LOCKED / AUTO / LOST state machine
      core.py                ties the above together (no I/O)
      tello.py               direct Wi-Fi link to a Tello
      app.py, dashboard/     the asyncio app and the browser ground station
      sim/                   the simulator (world, drone, camera)
      replay.py, logger.py   log every flight, re-run it later
    tests/                   148 tests
    config.yaml              all settings;  config.tello.yaml  Tello overrides
  training/                  rebuild the aerial-wildlife detector from public datasets (README inside)
  android-bridge/            optional DJI Mobile SDK v5 phone relay (parked, see below)
```

---

## 1. Run the simulation (no drone needed)

The simulator renders a top-down camera view of a simulated elephant walking among cattle and trees, and a simulated
drone you can hand over to the real autopilot code.

**Requirements:** Windows, macOS or Linux, **Python 3.11**, about 1.5 GB of free RAM. No GPU needed.

```bash
git clone https://github.com/AbhishMB/WildTrack.git
cd WildTrack/laptop

python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Live demo with the dashboard:**

```bash
python -m wildtrack sim
```

Open **http://localhost:8000**, then:

1. Wait about 2 seconds until the elephant's box turns **amber** (a lockable candidate).
2. **Click the box** (or press **L**) to lock it. The banner turns **LOCKED**.
3. Press **E** (ENGAGE). The banner turns green, **AUTO**: the drone follows the elephant. The map shows the drone,
   the animal's position with its error circle, and the trail.
4. Try **Pilot presses Pause** (sim panel) or **Esc** to see control handed back to the pilot.

**Numbers only (no browser):** runs in virtual time, much faster than real time, and prints how well the drone
followed:

```bash
python -m wildtrack sim --headless --plot run.png
python -m wildtrack sim --headless --speed 3.0 --tau 0.8     # faster animal, more delay
python -m wildtrack sim --headless --hide 20 24 --trees      # animal hidden for 4 s
```

**Run the tests:**

```bash
pytest tests
```

**What is simulated, and what is real:** the world, the drone's response and the camera are simulated (with video lag,
command delay, detection noise, dropouts and occlusions). The detector in the live demo reads the simulator's
ground truth. Everything else (locking rules, geolocation maths, Kalman filter, PID controller, mode state machine,
dashboard, logging and replay) is the same code that flies a real drone.

---

## 2. Run it with a real drone

### What drone do I need?

The laptop controls the drone directly, so the drone must let a laptop send it commands and receive its video
over Wi-Fi. This project supports:

- **DJI/Ryze Tello** (including **Tello EDU**): the supported, documented route.
- Any other drone only if you write a link for it (see below).

Consumer DJI drones (Mini, Air, Mavic) can only be commanded through DJI's Android Mobile SDK, not directly from a
laptop. An experimental phone-relay route for those exists in `android-bridge/`, but it is **parked, has never
been compiled**, and is not part of the supported setup.

### What you need

- A Tello with a charged battery, and a laptop with Wi-Fi (it will join the Tello's own network, so it will have
  **no internet** while flying; install everything beforehand).
- **Python 3.11.** An NVIDIA GPU is strongly recommended for YOLO; the CPU works but is slow.
- Install the GPU build of PyTorch instead of the CPU one, e.g.
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`, then `pip install -r requirements.txt`.
- The first run downloads the YOLO weights (`yolo26s.pt`, COCO). COCO knows dogs, cats, horses, bears, elephants and
  similar animals, so a pet, a toy animal or a picture works as a stand-in target.

### Steps

All commands run from `WildTrack/laptop` with the virtual environment active, and use the Tello profile
`config.tello.yaml`.

1. **Prepare.** Fly outdoors in an open area or a large hall, with no people nearby. Put the Tello on flat ground.
   Note which way its nose points and set `tello.takeoff_heading_deg` (0 = north) in `config.yaml`.
2. **Connect.** Switch the Tello on and join the laptop to its `TELLO-xxxxxx` Wi-Fi.
3. **Bench check (nothing flies):**
   ```bash
   python -m wildtrack --config config.tello.yaml bench
   ```
   It checks the state packets (about 10 Hz), battery, height and video.
4. **Measure the video lag** (hold the drone and turn it left and right):
   ```bash
   python -m wildtrack --config config.tello.yaml calibrate-lag --write
   ```
5. **Start the ground station:**
   ```bash
   python -m wildtrack --config config.tello.yaml run
   ```
   Open **http://localhost:8000**.
6. **Fly manually.** Press **TAKEOFF**, then fly with **W A S D** (move), **R / F** (up / down), **, / .** (turn).
   First check that the drone moves the way the map says it does.
7. **Calibrate** until the map matches reality: `tello.rc_per_mps`, `tello.vg_forward_sign`, `tello.vg_right_sign`,
   `tello.camera_pitch_deg` and `camera.hfov_deg` (see the comments in `config.yaml`).
8. **Lock and follow.** Aim the camera at the animal. When its box turns amber, click it (or press **L**), then press
   **E** to engage. Start with a slow, predictable target.
9. **Take back control** at any moment with **Esc** (hover and return to MANUAL), **LAND**, or the
   **EMERGENCY STOP** button, which cuts the motors immediately (the drone will drop).
10. After the flight, every sortie is logged in `logs/`. Re-run it with
    `python -m wildtrack replay <log-folder>` and export the animal's track with
    `python -m wildtrack export <log-folder>` (GPX and KML).

### Using another drone

Implement a link class with the same three members as `TelloLink` in `laptop/wildtrack/tello.py`: `run()` (connect and
keep connected), `send(msg)` (accepts `vel`, `vs_enable`, `vs_disable`, ...) and `connected`. It must call
`on_telemetry` with the telemetry message described in `models.py`. Register it in `make_link()`.

---

## Safety and legal

- Operate the drone **only where it is legal and safe**: keep it in sight, away from people, and within your country's
  drone rules. In India, check the Drone Rules 2021 and the Digital Sky airspace map before flying.
- Flying near wildlife can disturb it. For real wildlife work, get **written permission from the relevant authority**
  (for example the forest department), and follow their distance and altitude rules.
- Autopilot features are experimental. Keep a hand on **Esc** and the emergency stop, and test in simulation first.

## Known limits

- **Calibrate per drone.** The speed scale (`rc_per_mps`), camera angle, field of view, video lag and axis signs
  depend on your drone and are set in `config.yaml` (see step 7 above).
- **A Tello has no GPS.** Its position is estimated from its own speed reports and drifts over time; coordinates are
  relative to the take-off point (shown as latitude/longitude around `tello.home`). Range is a few metres and a
  flight lasts a few minutes. The error radius shown on the dashboard grows to reflect this.
- **The detector needs training.** Until you train it, the app uses COCO weights, which know animals from the side,
  not from the air. [`training/`](training/README.md) covers the base model (YOLO26s), the public aerial datasets
  (WAID, BuckTales, the Aerial Elephant Dataset) and the three commands. Public aerial data covers elephants, blackbuck,
  cattle, sheep and others, but we found none with leopards or tigers; those need your own labelled footage.
- The map is blank unless you add offline map tiles (`dashboard.tiles_dir` in `config.yaml`).

## Licence

The code in this repository is released under the **MIT License** (see [LICENSE](LICENSE)).

It depends on **Ultralytics YOLO, which is licensed AGPL-3.0**. If you distribute or deploy a product that includes
Ultralytics, you must comply with the AGPL-3.0 (or obtain an Ultralytics Enterprise licence for commercial use).
