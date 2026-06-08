# Copyright (c) SJTU, Z.Wu.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/jone-Wu/LSP-SAM2_v2
import numpy as np
import cv2


class BoxKalman:
    """
    Using Kalman to perform dual constraints for box prediction on key frames
    """

    def __init__(self, dt=1.0):
        self.dt = dt
        self.kf = cv2.KalmanFilter(8, 4, 0)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        self._initialized = False

    def initialize(self, boxes_history):
        """
        Initialize the Kalman filter using the history boxes of the object on last 5 frames.

        return:
            bool: whether the initialization successful
        """
        if len(boxes_history) < 3:
            return False

        centers_sizes = []
        for box in boxes_history:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            centers_sizes.append([cx, cy, w, h])
        centers_sizes = np.array(centers_sizes, dtype=np.float32)

        last = centers_sizes[-1]
        if len(centers_sizes) >= 2:
            prev = centers_sizes[-2]
            vx = (last[0] - prev[0]) / self.dt
            vy = (last[1] - prev[1]) / self.dt
            vw = (last[2] - prev[2]) / self.dt
            vh = (last[3] - prev[3]) / self.dt
        else:
            vx = vy = vw = vh = 0.0

        self.kf.statePost = np.array([
            [last[0]], [last[1]], [last[2]], [last[3]],
            [vx], [vy], [vw], [vh]
        ], dtype=np.float32)

        if len(centers_sizes) >= 3:
            # 计算相邻帧差分
            diffs = np.diff(centers_sizes, axis=0)  # shape: (n-1, 4)
            pos_var = np.var(diffs, axis=0)  # [var_cx, var_cy, var_w, var_h]
        else:
            pos_var = np.array([4.0, 4.0, 2.0, 2.0])

        vel_var = pos_var * 0.25
        q_diag = np.concatenate([pos_var, vel_var])
        self.kf.processNoiseCov = np.diag(q_diag.astype(np.float32))

        if len(centers_sizes) >= 4:
            obs_var = np.var(diffs, axis=0) * 0.5
        else:
            obs_var = np.array([9.0, 9.0, 4.0, 4.0])
        self.kf.measurementNoiseCov = np.diag(obs_var.astype(np.float32))
        p_diag = np.concatenate([pos_var * 10, vel_var * 10])
        self.kf.errorCovPost = np.diag(p_diag.astype(np.float32))
        self._initialized = True
        return True

    def predict(self):
        """
        predict the box on the key frame。

        return:
            tuple: (x1, y1, x2, y2) else None
        """
        if not self._initialized:
            return None

        state_pred = self.kf.predict()
        cx, cy, w, h = state_pred[0, 0], state_pred[1, 0], state_pred[2, 0], state_pred[3, 0]
        return abs(cx), abs(cy), abs(w), abs(h)

    def update(self, model_box):
        if not self._initialized:
            return model_box

        x1, y1, x2, y2 = model_box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        state_corrected = self.kf.correct(measurement)

        cx_c, cy_c, w_c, h_c = (
            state_corrected[0, 0], state_corrected[1, 0],
            state_corrected[2, 0], state_corrected[3, 0]
        )
        w_c = max(w_c, 1.0)
        h_c = max(h_c, 1.0)

        x1_c = cx_c - w_c / 2.0
        y1_c = cy_c - h_c / 2.0
        x2_c = cx_c + w_c / 2.0
        y2_c = cy_c + h_c / 2.0

        return (x1_c, y1_c, x2_c, y2_c)

    def smooth(self, model_box):
        """
        predict & update。

        parameter:
            model_box: [x1, y1, x2, y2]
        return:
            tuple: (x1, y1, x2, y2) 修正框
        """
        self.predict()
        return self.update(model_box)