"""
Angle utilities including 2D rotation matrix and angle normalization

author: Kat-yuan-eng (RuiWen Liao)
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle.

    :param angle: (float) Rotation angle in radians.
    :return: (numpy.ndarray) A 2D rotation matrix of shape (2, 2).
    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation. Default angle modulo range is [-pi, pi).

    :param x: (float or array_like) A angle or an array of angles. This array is flattened for the calculation.
    :param zero_2_2pi: (bool) Change angle modulo range to [0, 2pi). Default is False.
    :param degree: (bool) If True, the given angles are assumed to be in degrees. Default is False.
    :return: (float or ndarray) An angle or an array of modulated angle.
    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


if __name__ == '__main__':
    print("=== rot_mat_2d tests ===")
    print("rot_mat_2d(0):")
    print(rot_mat_2d(0.0))
    print("rot_mat_2d(pi/2):")
    print(rot_mat_2d(np.pi / 2.0))
    print("rot_mat_2d(pi):")
    print(rot_mat_2d(np.pi))

    print("\n=== angle_mod tests ===")
    print(f"angle_mod(-4.0)       = {angle_mod(-4.0):.8f}")
    print(f"angle_mod([-4.0])     = {angle_mod([-4.0])}")
    print(f"angle_mod([-150, 190, 350], degree=True) = {angle_mod([-150.0, 190.0, 350], degree=True)}")
    print(f"angle_mod(-60.0, zero_2_2pi=True, degree=True) = {angle_mod(-60.0, zero_2_2pi=True, degree=True)}")
    print(f"angle_mod(3*pi)       = {angle_mod(3 * np.pi):.8f}")
    print(f"angle_mod(-3*pi)      = {angle_mod(-3 * np.pi):.8f}")
