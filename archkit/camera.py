"""Camera models and photo camera-matching (for verified photomontages).

`solve_camera` recovers a real photo's camera (position, orientation, focal
length) from >= 6 control points whose site coordinates are known (survey
points, building corners from the site model, kerb lines...).  It uses the
normalised Direct Linear Transform followed by Gauss-Newton refinement of a
pinhole model with square pixels and centred principal point.  Reprojection
RMS (pixels) is reported so the match quality can be stated on the drawing,
as required for Accurate Visual Representations (AVR / verified views).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    eye: np.ndarray           # (3,) site frame
    R: np.ndarray             # (3,3) world->camera rotation; camera looks down +z_cam
    focal_px: float
    width: int
    height: int
    sensor_mm: float = 36.0

    @property
    def focal_mm(self) -> float:
        return self.focal_px * self.sensor_mm / self.width

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (n,3) -> pixel coords (n,2) and camera depth (n,)."""
        pc = (np.asarray(pts, float) - self.eye) @ self.R.T
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.width / 2 + self.focal_px * pc[:, 0] / z
            v = self.height / 2 - self.focal_px * pc[:, 1] / z
        return np.stack([u, v], 1), z

    @property
    def forward(self) -> np.ndarray:
        return self.R[2]

    @property
    def up(self) -> np.ndarray:
        return self.R[1]


def look_at(eye, target, focal_mm, sensor_mm, width, height, up=(0, 0, 1)) -> Camera:
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, u, f])   # rows: camera x (right), y (up), z (forward)
    return Camera(eye, R, focal_mm * width / sensor_mm, width, height, sensor_mm)


def _normalize(pts):
    c = pts.mean(0)
    d = np.sqrt(((pts - c) ** 2).sum(1)).mean()
    s = math.sqrt(pts.shape[1]) / (d if d > 0 else 1)
    T = np.eye(pts.shape[1] + 1)
    T[:-1, :-1] *= s
    T[:-1, -1] = -s * c
    return T


def _rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def solve_camera(px: np.ndarray, world: np.ndarray, width: int, height: int,
                 sensor_mm: float = 36.0) -> tuple[Camera, float]:
    """Solve camera from pixel (n,2) <-> world (n,3) correspondences. Returns (camera, rms_px)."""
    px, world = np.asarray(px, float), np.asarray(world, float)
    n = len(px)
    if n < 6:
        raise ValueError("camera solve needs at least 6 control points")
    # Pixel -> centred, y-up coordinates.
    uv = np.stack([px[:, 0] - width / 2, height / 2 - px[:, 1]], 1)
    Tu, Tx = _normalize(uv), _normalize(world)
    uvn = (Tu @ np.c_[uv, np.ones(n)].T).T[:, :2]
    Xn = (Tx @ np.c_[world, np.ones(n)].T).T
    A = []
    for (u, v), X in zip(uvn, Xn):
        A.append(np.r_[X, np.zeros(4), -u * X])
        A.append(np.r_[np.zeros(4), X, -v * X])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    P = np.linalg.inv(Tu) @ Vt[-1].reshape(3, 4) @ Tx
    # Decompose P = K [R | -R C].  Fix the projective scale sign so det(M) > 0,
    # which (with positive-diagonal K) puts the scene in front of the camera.
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    M = P[:, :3]
    C = -np.linalg.solve(M, P[:, 3])
    E = np.flipud(np.eye(3))
    Q, Rr = np.linalg.qr((E @ M).T)
    K = E @ Rr.T @ E
    Rm = E @ Q.T
    D = np.diag(np.sign(np.diag(K)))
    K, Rm = K @ D, D @ Rm
    f0 = abs((K[0, 0] + K[1, 1]) / 2 / K[2, 2])
    cam = Camera(C, Rm, f0, width, height, sensor_mm)
    # Cheirality: negating all rows of R leaves pixels unchanged but flips depth.
    if np.median(cam.project(world)[1]) < 0:
        cam.R = -cam.R
    # Gauss-Newton refinement on (rotation vector, eye, focal).
    params = np.r_[np.zeros(3), cam.eye, cam.focal_px]
    R0 = cam.R.copy()

    def residual(p):
        c = Camera(p[3:6], _rodrigues(p[:3]) @ R0, p[6], width, height, sensor_mm)
        pr, _ = c.project(world)
        return (pr - px).ravel()

    for _ in range(50):
        r = residual(params)
        J = np.empty((len(r), 7))
        for i in range(7):
            h = 1e-6 * max(1.0, abs(params[i]))
            dp = params.copy()
            dp[i] += h
            J[:, i] = (residual(dp) - r) / h
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        params += step
        if np.linalg.norm(step) < 1e-10:
            break
    cam = Camera(params[3:6], _rodrigues(params[:3]) @ R0, params[6], width, height, sensor_mm)
    rms = float(np.sqrt(np.mean((residual(params).reshape(-1, 2) ** 2).sum(1))))
    return cam, rms


def camera_euler_blender(cam: Camera) -> tuple[float, float, float]:
    """XYZ Euler (radians) for a Blender camera (looks down local -Z, +Y up)."""
    # Blender camera local axes in world: x = right, y = up, -z = forward
    Rb = np.stack([cam.R[0], cam.R[1], -cam.R[2]], 1)  # columns
    sy = -Rb[2, 0]
    y = math.asin(max(-1, min(1, sy)))
    if abs(math.cos(y)) > 1e-6:
        x = math.atan2(Rb[2, 1], Rb[2, 2])
        z = math.atan2(Rb[1, 0], Rb[0, 0])
    else:
        x = math.atan2(-Rb[1, 2], Rb[1, 1])
        z = 0.0
    return x, y, z
