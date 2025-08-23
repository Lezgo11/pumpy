import numpy as np
import meshio

__all__ = ["volume_from_mesh"]

def _raw_volume(mesh) -> float:
    pts = mesh.points
    cells = mesh.cells_dict
    if "tetra" in cells:
        vol = 0.0
        for tet in cells["tetra"]:
            p0, p1, p2, p3 = pts[tet]
            vol += abs(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0)) / 6.0
        return float(vol)  # in native units^3
    if "triangle" in cells:
        vol6 = 0.0
        for tri in cells["triangle"]:
            p0, p1, p2 = pts[tri]
            vol6 += np.dot(np.cross(p0, p1), p2)
        return abs(float(vol6)) / 6.0  # in native units^3
    raise ValueError("No tetra/triangle cells found")


def volume_from_mesh(path: str, units: str = "auto") -> float:
    """
    Return chamber volume in **m³**.
    units: "m", "mm", "cm", or "auto" (default).
    - "auto": heuristics on raw volume (native units³):
        * > 1e3  -> assume mm³  -> scale 1e-9
        * 1..1e3 -> assume cm³  -> scale 1e-6
        * else   -> assume m³
    """
    m = meshio.read(path)
    v_native = _raw_volume(m)

    if units == "m":
        scale = 1.0
    elif units == "mm":
        scale = 1e-9
    elif units == "cm":
        scale = 1e-6
    else:  # auto
        if v_native > 1e3:       # typical heart ~1.2e5 in mm³
            scale = 1e-9
        elif v_native > 1.0:     # typical heart ~120 in cm³
            scale = 1e-6
        else:                    # already m³
            scale = 1.0
    return v_native * scale