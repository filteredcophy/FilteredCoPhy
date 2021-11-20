import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

KEYPOINTS_PRESENCE_THRESH = 0.1  # If alpha_5 above the threshold, the keypoint is still active
DISTANCE_THRESH = 10


class Memory:
    def __init__(self):
        self.matches = []
        self.misses = []
        self.false_positive = []
        self.num_swap = 0

    def __repr__(self):
        s = f"Matches : {self.matches}\nMissed : {self.misses}\nFalse P : {self.false_positive}"
        return s


class MOTAccumulator:
    def __init__(self):
        self.M = Memory()
        self.logs = []
        self.n_frame = 0

    def update(self, objects, keypoints, activities=None):
        """
        Args:
            objects: N, 2
            keypoints: K, 2
            activities: K, 1
        Returns:

        """
        assert len(objects.shape) == len(keypoints.shape)
        assert len(objects.shape) == 2

        self.n_frame += 1
        if activities is None:
            activities = np.ones_like(keypoints)[:, 0]

        ## STEP 1 : Find correspondances in Mt-1 still valid in Mt
        next_M = Memory()
        for match in self.M.matches:
            valid, distance = match.is_valid(objects, keypoints, activities)
            if valid:
                next_M.matches.append(Match(match.idx_object, match.idx_keypoint, distance))

        ## STEP 2 : Try to find new match
        objects_id, keypoints_id = self.find_unpaired(next_M, objects, keypoints)
        distance_matrix = np.inf * np.ones((len(objects_id), len(keypoints_id)))
        if len(objects_id) > 0:
            for i in range(len(objects_id)):
                for j in range(len(keypoints_id)):
                    ii, jj = objects_id[i], keypoints_id[j]
                    distance_matrix[i, j] = np.sqrt(((objects[ii, :2] - keypoints[jj, :2]) ** 2).sum())

            paired_objects_id, paired_keypoints_id = linear_sum_assignment(distance_matrix)
            possible_matches = [Match(objects_id[i], keypoints_id[j], distance_matrix[i, j]) for (i, j) in
                                zip(paired_objects_id, paired_keypoints_id)]

            next_M.num_swap = self.count_swap(possible_matches)
            for m in possible_matches:
                valid, distance = m.is_valid(objects, keypoints, activities)
                if valid:
                    next_M.matches.append(m)

        objects_id, keypoints_id = self.find_unpaired(next_M, objects, keypoints)
        for o in objects_id:
            next_M.misses.append(o)
        for k in keypoints_id:
            next_M.false_positive.append(k)

        self.logs.append(next_M)
        self.M = next_M

    def find_unpaired(self, next_M, objects, keypoints):
        paired_objects = [m.idx_object for m in next_M.matches]
        paired_keypoints = [m.idx_keypoint for m in next_M.matches]

        unpaired_objects = [k for k in range(objects.shape[0]) if k not in paired_objects]
        unpaired_keypoints = [k for k in range(keypoints.shape[0]) if k not in paired_keypoints]
        return unpaired_objects, unpaired_keypoints

    def count_swap(self, possible_match):
        swap_count = 0
        for next_match in possible_match:
            for match in self.M.matches:
                if next_match.idx_object == match.idx_object and next_match.idx_keypoint != match.idx_keypoint:
                    swap_count += 1
        return swap_count

    def compute_metrics(self):

        sum_distances = sum([sum([m.distance for m in l.matches]) for l in self.logs])
        sum_paired = sum([len(m.matches) for m in self.logs])
        if sum_paired != 0:
            motp = sum_distances / sum_paired
        else:
            motp = 0

        sum_error = sum([len(m.misses) + len(m.false_positive) + m.num_swap for m in self.logs])
        sum_objects = sum([len(m.misses) + len(m.matches) for m in self.logs])
        mota = 1 - sum_error / sum_objects

        return motp, mota


class Match:
    def __init__(self, idx_object, idx_keypoint, distance):
        self.idx_object = idx_object
        self.idx_keypoint = idx_keypoint
        self.distance = distance

    def is_valid(self, objects, keypoints, activities):
        o = objects[self.idx_object]
        k = keypoints[self.idx_keypoint]
        a = activities[self.idx_keypoint]

        distance = np.sqrt(((o[:2] - k[:2]) ** 2).sum())
        if distance < DISTANCE_THRESH and a > KEYPOINTS_PRESENCE_THRESH:
            return True, distance
        return False, np.nan

    def __repr__(self):
        return f"({self.idx_object}, {self.idx_keypoint})"


if __name__ == '__main__':
    states = np.load("states.npy")[0]
    keypoints = np.load("keypoints.npy")[0]

    activities = keypoints[..., 6]
    keypoints = keypoints[..., :2]

    acc = MOTAccumulator()
    for t in range(keypoints.shape[0]):
        acc.update(states[t], keypoints[t])

    print(acc.compute_metrics())
    for t, a in enumerate(acc.logs):

        for m in a.matches:
            plt.scatter(keypoints[t, m.idx_keypoint, 0], keypoints[t, m.idx_keypoint, 1], c="tab:green",
                        marker='v')
            plt.scatter(states[t, m.idx_object, 0], states[t, m.idx_object, 1], c='tab:blue', marker="x")
            plt.plot([keypoints[t, m.idx_keypoint, 0], states[t, m.idx_object, 0]],
                     [keypoints[t, m.idx_keypoint, 1], states[t, m.idx_object, 1]],
                     linewidth=1, c="black")
        for m in a.misses:
            plt.scatter(states[t, m, 0], states[t, m, 1], c='tab:red', marker="v", s=1)
        for m in a.false_positive:
            plt.scatter(keypoints[t, m, 0], keypoints[t, m, 1], c="tab:red", marker="x", s=1)
    plt.show()
