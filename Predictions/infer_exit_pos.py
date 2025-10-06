import numpy as np
from typing import Literal

IntersectionType = Literal["barrel", "top_cap", "bottom_cap", "none"]

def ray_cylinder_intersection_yaxis_batch(
  P: np.ndarray,
  D: np.ndarray,
  R: float = 307.5926 / 2.0,
  H: float = 271.4235 / 2.0,
  include_caps: bool = True,
  t_min: float = 0.0,
  eps: float = 1e-9,
):
  """
  Vectorized version: compute intersections for multiple rays with
  a y-axis aligned finite cylinder (centered at origin).

  Parameters
  ----------
  P : (N, 3) array
    Ray origins.
  D : (N, 3) array
    Ray directions (needn't to be normalized).
  R : float
    Cylinder radius.
  H : float
    Half height of cylinder.
  include_caps : bool
    Whether to include top/bottom caps.
  t_min : float
    Minimum valid ray parameter (usually 0).
  eps : float
    Small epsilon for numerical tolerance.

  Returns
  -------
  results : dict
    {
      "t": (N,), intersection distance or np.nan,
      "X": (N, 3), intersection point or nan,
      "kind": (N,), string category ("barrel"/"top_cap"/"bottom_cap"/"none")
    }
  """

  P = np.asarray(P, dtype=float)
  D = np.asarray(D, dtype=float)
  if P.ndim == 1:
    P = P[None, :]
    D = D[None, :]

  N = P.shape[0]

  # norm of D
  Dnorm = np.linalg.norm(D, axis=1)
  D = D / np.maximum(Dnorm, eps)[:, None]
  
  # unpack components
  Px, Py, Pz = P[:, 0], P[:, 1], P[:, 2]
  Dx, Dy, Dz = D[:, 0], D[:, 1], D[:, 2]


  # initialize outputs
  t_out = np.full(N, np.nan)
  X_out = np.full((N, 3), np.nan)
  kind_out = np.full(N, "none", dtype=object)

  # --- barrel surface ---
  A = Dx ** 2 + Dz ** 2
  B = 2 * (Px * Dx + Pz * Dz)
  C = Px ** 2 + Pz ** 2 - R ** 2
  delta = B ** 2 - 4 * A * C

  barrel_mask = (A > eps) & (delta >= 0)
  if np.any(barrel_mask):
    s = np.sqrt(np.maximum(0, delta[barrel_mask]))
    t_candidates = np.stack([
      (-B[barrel_mask] - s) / (2 * A[barrel_mask]),
      (-B[barrel_mask] + s) / (2 * A[barrel_mask])
    ], axis=1)

    # pick nearest valid t ≥ t_min within height range
    for i, idx in enumerate(np.where(barrel_mask)[0]):
      for t in np.sort(t_candidates[i]):
        if t >= t_min:
          y = Py[idx] + t * Dy[idx]
          if -H - eps <= y <= H + eps:
            t_out[idx] = t
            X_out[idx] = P[idx] + t * D[idx]
            kind_out[idx] = "barrel"
            break

  # --- caps ---
  if include_caps:
    for label, y_cap in [("top_cap", H), ("bottom_cap", -H)]:
      mask = np.abs(Dy) > eps
      tcap = (y_cap - Py) / Dy
      mask &= (tcap >= t_min)
      Xcap = P + tcap[:, None] * D
      inside = (Xcap[:, 0]**2 + Xcap[:, 2]**2 <= R**2 + eps)
      cap_mask = mask & inside & np.isnan(t_out)
      if np.any(cap_mask):
        t_out[cap_mask] = tcap[cap_mask]
        X_out[cap_mask] = Xcap[cap_mask]
        kind_out[cap_mask] = label

  return {"t": t_out, "X": X_out, "kind": kind_out}