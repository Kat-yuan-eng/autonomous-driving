"""Localization degradation detection and fallback management

author: Kat-yuan-eng (RuiWen Liao)
"""
# === Phase 1: Four-state degradation decision ===
import numpy as np

from SLAM.config import SCORE_HEALTHY, SCORE_DEGRADE


class DegradationManager:

    STATUS_MAP = {0: 'normal', 1: 'carto_degraded'}

    def __init__(self):
        self.status = 0
        self.status_log = []
        self.last_valid_pose = None

    def decide(self, carto_health):
        assert isinstance(carto_health, dict), "carto_health must be dict"
        carto_ok = carto_health.get('healthy', False)
        carto_degraded = carto_health.get('degraded', False)
        if carto_ok:
            new_status = 0
        else:
            new_status = 1
        if new_status != self.status:
            self.status_log.append((len(self.status_log), new_status))
        self.status = new_status
        return self.STATUS_MAP[new_status]

    def get_fused_pose(self, ukf_pose, carto_pose):
        if self.status == 0:
            pose = ukf_pose.copy()
        else:
            pose = ukf_pose.copy()
        if not np.all(np.isfinite(pose)):
            if self.last_valid_pose is not None:
                pose = self.last_valid_pose.copy()
            else:
                pose = np.zeros(3)
        self.last_valid_pose = pose.copy()
        return pose

    def get_emergency_stop(self):
        return False
