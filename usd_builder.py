"""Stage 2: turn a Scene into a USD stage."""

import json
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdLux

from scene_extractor import Actor, Scene

# Ego at origin, +Z = forward, +X = right, +Y = up.
LANE_WIDTH_M = 3.5
EYE_HEIGHT_M = 1.25

DISTANCE_M = {"very_close": 4.0, "close": 10.0, "medium": 25.0, "far": 55.0}

LANE_LAYOUT = {
    "same_ahead":     (0.0,                 +1),
    "same_behind":    (0.0,                 -1),
    "oncoming_ahead": (-LANE_WIDTH_M,       +1),
    "oncoming_left":  (-LANE_WIDTH_M,       +1),
    "left_shoulder":  (-LANE_WIDTH_M * 1.5, +1),
    "right_shoulder": (+LANE_WIDTH_M * 1.5, +1),
    "parked_left":    (-LANE_WIDTH_M * 1.5, +1),
    "parked_right":   (+LANE_WIDTH_M * 1.5, +1),
    "crosswalk":      (0.0,                 +1),
}

VEHICLE_DIMS = {
    "car":        (1.8, 1.5, 4.5),
    "truck":      (2.5, 3.2, 8.0),
    "van":        (2.0, 2.2, 5.5),
    "bus":        (2.5, 3.2, 12.0),
    "motorcycle": (0.8, 1.3, 2.2),
}

COLOR_MAP = {
    "red":    (0.75, 0.10, 0.10),
    "orange": (0.90, 0.45, 0.10),
    "yellow": (0.90, 0.85, 0.10),
    "green":  (0.15, 0.55, 0.20),
    "blue":   (0.10, 0.25, 0.75),
    "black":  (0.06, 0.06, 0.07),
    "white":  (0.92, 0.92, 0.92),
    "silver": (0.75, 0.75, 0.77),
    "gray":   (0.45, 0.45, 0.47),
    "grey":   (0.45, 0.45, 0.47),
    "brown":  (0.35, 0.20, 0.10),
    "dark":   (0.15, 0.15, 0.17),
}

WEATHER_LIGHT_COLOR = {
    "clear":    (1.00, 0.95, 0.85),
    "overcast": (0.75, 0.78, 0.82),
    "rain":     (0.55, 0.60, 0.68),
    "snow":     (0.90, 0.92, 0.96),
    "fog":      (0.72, 0.72, 0.72),
}
WEATHER_LIGHT_INTENSITY = {
    "clear": 4500, "overcast": 2200, "rain": 1500, "snow": 3000, "fog": 1800,
}

WHEEL_COLOR = (0.07, 0.07, 0.07)
WINDSHIELD_COLOR = (0.12, 0.18, 0.28)
CURB_COLOR = (0.72, 0.72, 0.72)
SIDEWALK_COLOR = (0.62, 0.62, 0.62)
LAWN_COLOR = (0.20, 0.35, 0.15)
CENTERLINE_COLOR = (0.95, 0.85, 0.15)


def _color_for(name: str) -> tuple[float, float, float]:
    key = (name or "gray").lower().strip()
    for token in key.split():
        if token in COLOR_MAP:
            return COLOR_MAP[token]
    return COLOR_MAP["gray"]


def _darken(color, factor=0.7):
    return tuple(c * factor for c in color)


def _apply_color(prim, rgb) -> None:
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])


def _make_cube(stage, path, size_xyz, center_xyz, color, rot_y_deg=0.0):
    prim = UsdGeom.Cube.Define(stage, path)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center_xyz))
    if rot_y_deg:
        xf.AddRotateYOp().Set(rot_y_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(size_xyz[0] / 2, size_xyz[1] / 2, size_xyz[2] / 2))
    _apply_color(prim.GetPrim(), color)


def _make_wheel(stage, path, center, radius=0.32, thickness=0.20):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(thickness)
    cyl.CreateAxisAttr("X")
    _apply_color(cyl.GetPrim(), WHEEL_COLOR)


def _add_vehicle(stage, base_path, dims, ground_xz, heading_deg, body_color):
    """Composite car: chassis + cabin + windshield + 4 wheels. Sits on Y=0."""
    W, H, L = dims
    x, z = ground_xz

    group = UsdGeom.Xform.Define(stage, base_path)
    gxf = UsdGeom.Xformable(group)
    gxf.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, z))
    if heading_deg:
        gxf.AddRotateYOp().Set(heading_deg)

    body_h = H * 0.55
    cabin_h = H * 0.45
    cabin_w = W * 0.92
    cabin_l = L * 0.55
    cabin_z = -L * 0.05  # cabin biased slightly rearward

    _make_cube(stage, f"{base_path}/Body", (W, body_h, L),
               (0, body_h / 2, 0), body_color)
    _make_cube(stage, f"{base_path}/Cabin", (cabin_w, cabin_h, cabin_l),
               (0, body_h + cabin_h / 2, cabin_z), _darken(body_color, 0.75))
    _make_cube(stage, f"{base_path}/Windshield",
               (cabin_w * 0.95, cabin_h * 0.85, 0.08),
               (0, body_h + cabin_h / 2, cabin_z + cabin_l / 2 + 0.04),
               WINDSHIELD_COLOR)

    wheel_r = min(0.36, H * 0.25)
    wheel_x = W / 2 - 0.05
    wheel_z_front = L / 2 - 0.75
    wheel_z_rear = -L / 2 + 0.75
    for wx, wz, tag in [
        (-wheel_x, wheel_z_front, "FL"),
        (+wheel_x, wheel_z_front, "FR"),
        (-wheel_x, wheel_z_rear,  "RL"),
        (+wheel_x, wheel_z_rear,  "RR"),
    ]:
        _make_wheel(stage, f"{base_path}/Wheel_{tag}",
                    (wx, wheel_r, wz), radius=wheel_r)


def _add_cyclist(stage, base_path, ground_xz, color):
    x, z = ground_xz
    group = UsdGeom.Xform.Define(stage, base_path)
    UsdGeom.Xformable(group).AddTranslateOp().Set(Gf.Vec3d(x, 0.0, z))
    _make_cube(stage, f"{base_path}/Torso", (0.45, 0.75, 0.28), (0, 1.15, 0), color)
    head = UsdGeom.Sphere.Define(stage, f"{base_path}/Head")
    UsdGeom.Xformable(head).AddTranslateOp().Set(Gf.Vec3d(0, 1.72, 0))
    head.CreateRadiusAttr(0.13)
    _apply_color(head.GetPrim(), _darken(color, 0.8))
    _make_cube(stage, f"{base_path}/Frame", (0.05, 0.55, 1.1), (0, 0.55, 0), (0.20, 0.20, 0.20))
    _make_wheel(stage, f"{base_path}/Wheel_F", (0, 0.35, +0.5), radius=0.35, thickness=0.05)
    _make_wheel(stage, f"{base_path}/Wheel_R", (0, 0.35, -0.5), radius=0.35, thickness=0.05)


def _add_pedestrian(stage, base_path, ground_xz, color):
    x, z = ground_xz
    group = UsdGeom.Xform.Define(stage, base_path)
    UsdGeom.Xformable(group).AddTranslateOp().Set(Gf.Vec3d(x, 0.0, z))
    _make_cube(stage, f"{base_path}/Leg_L", (0.16, 0.85, 0.16), (-0.10, 0.425, 0), (0.15, 0.15, 0.20))
    _make_cube(stage, f"{base_path}/Leg_R", (0.16, 0.85, 0.16), (+0.10, 0.425, 0), (0.15, 0.15, 0.20))
    _make_cube(stage, f"{base_path}/Torso", (0.42, 0.55, 0.25), (0, 1.15, 0), color)
    head = UsdGeom.Sphere.Define(stage, f"{base_path}/Head")
    UsdGeom.Xformable(head).AddTranslateOp().Set(Gf.Vec3d(0, 1.60, 0))
    head.CreateRadiusAttr(0.12)
    _apply_color(head.GetPrim(), (0.80, 0.65, 0.55))


def _add_ground(stage, road_type):
    color = (0.22, 0.22, 0.24) if road_type != "parking" else (0.28, 0.28, 0.30)
    _make_cube(stage, "/World/Ground", (100.0, 0.10, 240.0), (0, -0.05, 0), color)
    _make_cube(stage, "/World/Lawn_L", (46, 0.05, 240.0), (-27, -0.020, 0), LAWN_COLOR)
    _make_cube(stage, "/World/Lawn_R", (46, 0.05, 240.0), (+27, -0.020, 0), LAWN_COLOR)


def _add_sidewalks_and_curbs(stage, road_type):
    if road_type == "parking":
        return
    curb_x = LANE_WIDTH_M + 0.10
    _make_cube(stage, "/World/Curb_L", (0.15, 0.15, 240.0), (-curb_x, 0.075, 0), CURB_COLOR)
    _make_cube(stage, "/World/Curb_R", (0.15, 0.15, 240.0), (+curb_x, 0.075, 0), CURB_COLOR)
    sw_x = curb_x + 1.25 + 0.075
    _make_cube(stage, "/World/Sidewalk_L", (2.5, 0.10, 240.0), (-sw_x, 0.05, 0), SIDEWALK_COLOR)
    _make_cube(stage, "/World/Sidewalk_R", (2.5, 0.10, 240.0), (+sw_x, 0.05, 0), SIDEWALK_COLOR)


def _add_lane_stripes(stage, lanes, road_type):
    if road_type == "parking":
        return
    if lanes >= 2 and road_type in ("residential", "urban"):
        _make_cube(stage, "/World/CenterLine_A", (0.10, 0.02, 240.0), (-0.10, 0.015, 0), CENTERLINE_COLOR)
        _make_cube(stage, "/World/CenterLine_B", (0.10, 0.02, 240.0), (+0.10, 0.015, 0), CENTERLINE_COLOR)
        return
    stripe_half_len = 1.5
    gap = 4.0
    z = -100.0
    idx = 0
    while z < 100.0:
        _make_cube(
            stage,
            f"/World/Stripes/Stripe_{idx:03d}",
            (0.15, 0.02, stripe_half_len * 2),
            (0.0, 0.015, z + stripe_half_len),
            (0.95, 0.95, 0.95),
        )
        z += stripe_half_len * 2 + gap
        idx += 1


def _add_ego(stage):
    _add_vehicle(stage, "/World/Ego", VEHICLE_DIMS["car"], (0, 0),
                 heading_deg=0.0, body_color=(0.78, 0.10, 0.10))
    # DashCam at root so it doesn't inherit any parent scale.
    # USD cameras look down -Z by default; rotate 180 around Y to face +Z.
    cam = UsdGeom.Camera.Define(stage, "/World/DashCam")
    cxf = UsdGeom.Xformable(cam)
    cxf.AddTranslateOp().Set(Gf.Vec3d(0.0, EYE_HEIGHT_M, VEHICLE_DIMS["car"][2] / 2 - 0.5))
    cxf.AddRotateYOp().Set(180.0)
    cam.CreateFocalLengthAttr(24.0)


def _place_actor(stage, actor: Actor, index: int, parked_index: int | None = None) -> None:
    x_off, z_dir = LANE_LAYOUT.get(actor.lane, (0.0, 1))
    z = DISTANCE_M[actor.distance_qualitative] * z_dir
    if actor.lane.startswith("parked_") and parked_index is not None:
        # Space parked cars along Z using their position among *parked-only* actors,
        # not the global actor index — otherwise non-parked actors gap the row and
        # push a later parked car onto z=0 (colliding with the ego).
        z = -30.0 + parked_index * 6.0

    color = _color_for(actor.color)
    path = f"/World/Actors/Actor_{index:03d}_{actor.type}"

    if actor.type == "pedestrian":
        _add_pedestrian(stage, path, (x_off, z), color)
    elif actor.type in ("cyclist", "motorcycle"):
        _add_cyclist(stage, path, (x_off, z), color)
    elif actor.type in VEHICLE_DIMS:
        heading = 180.0 if actor.lane.startswith("oncoming") else 0.0
        _add_vehicle(stage, path, VEHICLE_DIMS[actor.type], (x_off, z), heading, color)
    else:
        _add_vehicle(stage, path, VEHICLE_DIMS["car"], (x_off, z), 0.0, color)


def _add_environment_features(stage, features) -> None:
    for f in features:
        if f.type != "trees":
            continue
        sides = [-1, 1] if f.side == "both" else ([-1] if f.side == "left" else [1])
        for side in sides:
            side_tag = "L" if side == -1 else "R"
            base_x = side * (LANE_WIDTH_M * 2.5)
            for i, z in enumerate(range(-80, 81, 12)):
                jitter_x = ((i * 7919) % 100 - 50) / 30.0
                jitter_z = ((i * 3931) % 100 - 50) / 20.0
                x = base_x + jitter_x
                zj = float(z) + jitter_z

                trunk = UsdGeom.Cylinder.Define(stage, f"/World/Trees/Trunk_{side_tag}_{i:03d}")
                UsdGeom.Xformable(trunk).AddTranslateOp().Set(Gf.Vec3d(x, 1.5, zj))
                trunk.CreateRadiusAttr(0.2)
                trunk.CreateHeightAttr(3.0)
                trunk.CreateAxisAttr("Y")
                _apply_color(trunk.GetPrim(), COLOR_MAP["brown"])

                crown_r = 1.6 + ((i * 6151) % 100) / 250.0
                crown = UsdGeom.Sphere.Define(stage, f"/World/Trees/Crown_{side_tag}_{i:03d}")
                UsdGeom.Xformable(crown).AddTranslateOp().Set(Gf.Vec3d(x, 3.5 + crown_r * 0.5, zj))
                crown.CreateRadiusAttr(crown_r)
                _apply_color(crown.GetPrim(), COLOR_MAP["green"])


def _add_lighting(stage, scene: Scene) -> None:
    sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
    weather = scene.environment.weather
    color = WEATHER_LIGHT_COLOR.get(weather, (0.85, 0.85, 0.90))
    intensity = WEATHER_LIGHT_INTENSITY.get(weather, 2500)
    if scene.environment.time_of_day == "night":
        color = (0.05, 0.06, 0.12)
        intensity = 400
    elif scene.environment.time_of_day in ("dusk", "dawn"):
        color = (0.75, 0.55, 0.45)
        intensity = 1800
    sky.CreateColorAttr(Gf.Vec3f(*color))
    sky.CreateIntensityAttr(intensity * 0.05)

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(intensity)
    sun.CreateColorAttr(Gf.Vec3f(*color))
    elev = 55.0 if scene.environment.time_of_day == "day" else 15.0
    xform = UsdGeom.Xformable(sun)
    xform.AddRotateYOp().Set(float(scene.environment.sun_azimuth_deg))
    xform.AddRotateXOp().Set(-elev)


def build_stage(scene: Scene, out_path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    world.GetPrim().SetCustomDataByKey("sceneJson", json.dumps(scene.model_dump(), indent=2))
    world.GetPrim().SetCustomDataByKey("summary", scene.summary)
    stage.SetDefaultPrim(world.GetPrim())

    _add_ground(stage, scene.environment.road_type)
    _add_sidewalks_and_curbs(stage, scene.environment.road_type)
    _add_lane_stripes(stage, scene.environment.lanes, scene.environment.road_type)
    _add_ego(stage)
    parked_seen = 0
    for i, actor in enumerate(scene.actors):
        parked_i = None
        if actor.lane.startswith("parked_"):
            parked_i = parked_seen
            parked_seen += 1
        _place_actor(stage, actor, i, parked_i)
    _add_environment_features(stage, scene.environment_features)
    _add_lighting(stage, scene)

    stage.GetRootLayer().Save()
    return out_path
