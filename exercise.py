import util
import numpy as np
from typing import List, Optional


class ReferenceExercise:
    """
    Represents a reference exercise used for comparison with user performances.
    Stores keyframe data of an exercise from relaxed to engaged position and back.

    Attributes:
        keypoint_keyframes (np.ndarray): Array of pose keypoints for each keyframe
        regions_of_interest (np.ndarray): Body regions relevant for this exercise
        name (str): Name identifier for the exercise
        engaged_position_index (int): Index of most engaged position in keyframes
        relaxed_position_index (int): Index of most relaxed position in keyframes
    """

    def __init__(self, exercise_name: Optional[str] = None) -> None:
        self.keypoint_keyframes: np.ndarray = np.load(f"exercises/{exercise_name}/keyframes.npy")
        self.regions_of_interest: np.ndarray = np.load(f"exercises/{exercise_name}/roi.npy")
        self.name: str = exercise_name

        # Find the most extreme positions (engaged and relaxed) by calculating
        # distances between all pose keyframes
        pose_distances: List[List[float]] = []
        distance_count: List[int] = [0] * len(self.keypoint_keyframes)

        # Calculate distances between each pair of poses
        for i, pose_1 in enumerate(self.keypoint_keyframes):
            current_distances: List[float] = []
            for j, pose_2 in enumerate(self.keypoint_keyframes):
                if i != j:
                    distance = np.sum(util.calculate_pose_distance(pose_1, pose_2))
                else:
                    distance = 0.0
                current_distances.append(distance)

            pose_distances.append(current_distances)
            distance_count[np.argmax(current_distances)] += 1

        # Find the most extreme positions
        furthest_index = np.argmax(distance_count)
        opposite_index = np.argmax(pose_distances[furthest_index])

        # Assign engaged and relaxed positions
        if furthest_index > opposite_index:
            self.engaged_position_index = furthest_index
            self.relaxed_position_index = opposite_index
        else:
            self.engaged_position_index = opposite_index
            self.relaxed_position_index = furthest_index



