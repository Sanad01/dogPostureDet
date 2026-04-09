import numpy as np

def extract_features(kp):
    """
    kp: normalized keypoints (24, 2)
    """

    def dist(a, b):
        return np.linalg.norm(kp[a] - kp[b])

    def angle(a, b, c):
        # angle at b (a-b-c)
        ba = kp[a] - kp[b]
        bc = kp[c] - kp[b]

        cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    features = []

    # body structure
    NOSE = 16
    TAIL = 12

    features.append(dist(NOSE, TAIL))  # body length (should ~1 after normalization)

    # front legs
    FL_PAW = 0
    FL_ELBOW = 2

    FR_PAW = 6
    FR_ELBOW = 8

    # vertical distance between front body and back body
    elbow_y = kp[FL_ELBOW][1] + kp[FR_ELBOW][1] / 2
    features.append(elbow_y - kp[TAIL][1])

    # vertical differences
    features.append(kp[FL_PAW][1] - kp[FL_ELBOW][1])
    features.append(kp[FR_PAW][1] - kp[FR_ELBOW][1])

    # distances
    features.append(dist(FL_PAW, FL_ELBOW))
    features.append(dist(FR_PAW, FR_ELBOW))

    # back legs
    RL_PAW = 3
    RL_KNEE = 4

    RR_PAW = 9
    RR_KNEE = 10

    features.append(dist(RL_PAW, RL_KNEE))
    features.append(dist(RR_PAW, RR_KNEE))

    # ANGLES
    # leg bending
    features.append(angle(FL_PAW, FL_ELBOW, NOSE))
    features.append(angle(FR_PAW, FR_ELBOW, NOSE))

    features.append(angle(RL_PAW, RL_KNEE, TAIL))
    features.append(angle(RR_PAW, RR_KNEE, TAIL))

    # vertical spread
    y_range = np.max(kp[:, 1]) - np.min(kp[:, 1])
    features.append(y_range)

    # difference in vertical position of front and back paws
    front_y = (kp[FL_PAW][1] + kp[FR_PAW][1]) / 2
    back_y = (kp[RL_PAW][1] + kp[RR_PAW][1]) / 2

    features.append(abs(front_y - back_y))

    # spine shape
    MID = (kp[NOSE] + kp[TAIL]) / 2

    # measure "curvature"
    features.append(np.linalg.norm(kp[NOSE] - MID) - np.linalg.norm(kp[TAIL] - MID))

    return np.array(features)
