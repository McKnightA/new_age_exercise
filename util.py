import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import List, Dict, Tuple, Optional, Any
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from exercise import ReferenceExercise


def get_pose_connections() -> List[List[int]]:
    """
    Returns the connections between body keypoints in the MediaPipe pose model.
    Each index represents a keypoint, and its value is a list of connected keypoints.

    Returns:
        List[List[int]]: Connection indices for the pose skeleton
    """
    return [[2, 5], [2], [0, 1, 3, 7], [3], [5], [0, 4, 6, 8],
            [5], [2], [5], [10], [9], [12, 13, 23], [11, 14, 24],
            [11, 15], [12, 16], [13, 17, 21], [14, 18, 22], [15, 19],
            [16, 20], [17, 15], [16, 18], [15], [16], [11, 24, 25],
            [12, 23, 26], [23, 27], [24, 28], [25, 29, 31], [26, 30, 32],
            [27, 29], [28, 32], [27, 31], [28, 30]]


def get_body_segments() -> Dict[str, List[int]]:
    """
    Returns a dictionary mapping body segment names to their constituent keypoint indices.

    Returns:
        Dict[str, List[int]]: Mapping of body segment names to keypoint indices
    """
    return {
        "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "torso": [11, 12, 23, 24],
        "left_arm": [11, 13, 15, 17, 19, 21],
        "right_arm": [12, 14, 16, 18, 20, 22],
        "left_leg": [23, 25, 27, 29, 31],
        "right_leg": [24, 26, 28, 30, 32]
    }


def reorient_pose_data(pose_data: np.ndarray) -> np.ndarray:
    """
    Reorients pose data to standardize the viewing angle.

    Args:
        pose_data (np.ndarray): Original pose keypoint data

    Returns:
        np.ndarray: Reoriented pose data
    """
    reoriented_data = pose_data.copy()
    z_coord = reoriented_data[:, 2].copy()
    reoriented_data[:, 2] = reoriented_data[:, 1] * -1
    reoriented_data[:, 1] = z_coord
    return reoriented_data


def calculate_rotation_matrix(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    Calculates the rotation matrix that aligns vector1 to vector2.

    Args:
        vector1 (np.ndarray): Source 3D vector
        vector2 (np.ndarray): Destination 3D vector

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Normalize vectors
    v1_normalized = (vector1 / np.linalg.norm(vector1)).reshape(3)
    v2_normalized = (vector2 / np.linalg.norm(vector2)).reshape(3)

    # Calculate cross product and dot product
    cross_product = np.cross(v1_normalized, v2_normalized)
    dot_product = np.dot(v1_normalized, v2_normalized)

    # Calculate skew-symmetric cross-product matrix
    cross_product_matrix = np.array([
        [0, -cross_product[2], cross_product[1]],
        [cross_product[2], 0, -cross_product[0]],
        [-cross_product[1], cross_product[0], 0]
    ])

    # Calculate rotation matrix using Rodrigues' rotation formula
    cross_magnitude = np.linalg.norm(cross_product)
    rotation_matrix = (np.eye(3) + cross_product_matrix +
                       cross_product_matrix.dot(cross_product_matrix) *
                       ((1 - dot_product) / (cross_magnitude ** 2)))

    return rotation_matrix


def convert_landmarks_to_numpy(landmark_frames: List[Any]) -> np.ndarray:
    """
    Converts MediaPipe landmark format to numpy array format.

    Args:
        landmark_frames (List[Any]): List of MediaPipe landmark objects

    Returns:
        np.ndarray: Array of shape (n_frames, 3, n_landmarks) containing landmark coordinates
    """
    frames_list = []

    for landmark_set in landmark_frames:
        x_coords, y_coords, z_coords = [], [], []

        for landmark in landmark_set:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
            z_coords.append(landmark.z)

        frames_list.append([x_coords, y_coords, z_coords])

    return np.array(frames_list)


def calculate_pose_distance(pose_1: np.ndarray,
                            pose_2: np.ndarray,
                            regions_of_interest: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculates the Euclidean distance between two poses, optionally for specific regions.

    Args:
        pose_1 (np.ndarray): First pose keypoints
        pose_2 (np.ndarray): Second pose keypoints
        regions_of_interest (Optional[np.ndarray]): Indices of keypoints to consider

    Returns:
        np.ndarray: Array of distances between corresponding keypoints
    """
    if regions_of_interest is None:
        return np.sqrt(np.sum(np.square(pose_1 - pose_2), axis=0))
    else:
        return np.sqrt(np.sum(np.square(
            pose_1[:, regions_of_interest] - pose_2[:, regions_of_interest]
        ), axis=0))


def initialize_camera(output_filename: str) -> Tuple[cv2.VideoCapture, cv2.VideoWriter]:
    """
    Initializes camera capture and video writer objects.

    Args:
        output_filename (str): Name of the output video file

    Returns:
        Tuple[cv2.VideoCapture, cv2.VideoWriter]: Camera and video writer objects
    """
    camera = cv2.VideoCapture(0)
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        f'results/{output_filename}.mp4',
        fourcc,
        20.0,
        (frame_width, frame_height)
    )

    return camera, video_writer


def initialize_pose_model() -> Tuple[Any, Any]:
    """
    Initializes the MediaPipe pose detection model with appropriate settings.

    Returns:
        Tuple[Any, Any]: Pose landmarker class and configuration options
    """
    model_path = 'models/pose_landmarker_full.task'

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )

    return PoseLandmarker, options


def record_exercise_video(output_filename: str) -> None:
    """
    Records video from webcam until user presses 'q'.

    Args:
        output_filename (str): Name of the output video file
    """
    camera, video_writer = initialize_camera(output_filename)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            video_writer.write(frame)
            cv2.imshow('Exercise Recording', frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        camera.release()
        video_writer.release()
        cv2.destroyAllWindows()


def visualize_pose_landmarks(
        rgb_image: np.ndarray,
        detection_result: Any
) -> np.ndarray:
    """
    Draws pose landmarks and connections on the input image.

    Args:
        rgb_image (np.ndarray): Input image in RGB format
        detection_result: MediaPipe pose detection result

    Returns:
        np.ndarray: Image with pose landmarks and connections drawn
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        # Convert landmarks to protocol buffer format
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z
            ) for landmark in pose_landmarks
        ])

        # Draw the pose landmarks
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    return annotated_image


def animate_keypoint_sequence(
        keypoint_frames: np.ndarray,
        output_filename: str,
        frame_times: Optional[np.ndarray] = None
) -> None:
    """
    Creates an animated visualization of pose keypoints over time.

    Args:
        keypoint_frames (np.ndarray): Sequence of pose keyframes
        output_filename (str): Name of the output animation file
        frame_times (Optional[np.ndarray]): Timestamps for each frame
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Reorient data for visualization
    reoriented_frames = reorient_pose_data(keypoint_frames.copy())
    pose_connections = get_pose_connections()

    def update_frame(frame_idx: int) -> None:
        """Updates the plot for each animation frame."""
        ax.clear()

        # Set consistent axes properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # Plot keypoints
        ax.scatter(
            reoriented_frames[frame_idx, 0],
            reoriented_frames[frame_idx, 1],
            reoriented_frames[frame_idx, 2]
        )

        # Draw connections between keypoints
        for i, connections in enumerate(pose_connections):
            for j in connections:
                if j > i:
                    ax.plot(
                        [reoriented_frames[frame_idx, 0, i],
                         reoriented_frames[frame_idx, 0, j]],
                        [reoriented_frames[frame_idx, 1, i],
                         reoriented_frames[frame_idx, 1, j]],
                        [reoriented_frames[frame_idx, 2, i],
                         reoriented_frames[frame_idx, 2, j]],
                        color='r'
                    )

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(reoriented_frames) - 1
    )

    writer = animation.FFMpegWriter(
        fps=5,
        extra_args=['-vcodec', 'libx264']
    )
    anim.save(f'animations/{output_filename}_3d.mp4', writer=writer)
    plt.close(fig)


def generate_reference_exercise(exercise_name: str, regions_of_interest: List[str]) -> None:
    """
    Generates a reference exercise by processing a video recording and extracting key poses.

    Args:
        exercise_name: Name of the exercise to generate reference data for
        regions_of_interest: List of body regions to focus on (e.g., ["torso", "left leg"])
    """
    # Create directory for exercise data if it doesn't exist
    video_path = f'results/{exercise_name}.mp4'
    exercise_dir = Path(f"exercises/{exercise_name}")
    exercise_dir.mkdir(parents=True, exist_ok=True)

    # Initialize video capture and pose detection model
    video_capture = cv2.VideoCapture(video_path)
    pose_landmarker, model_options = initialize_pose_model()

    keypoints = []
    timestamps = []

    # Process video frames and extract pose landmarks
    with pose_landmarker.create_from_options(model_options) as landmarker:
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            frame_time = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamps.append(frame_time)

            # Detect pose landmarks in the frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            pose_result = landmarker.detect_for_video(mp_image, frame_time)

            if pose_result.pose_world_landmarks:
                keypoints.append(pose_result.pose_world_landmarks[0])

    # Convert landmarks to numpy array for processing
    keypoints = convert_landmarks_to_numpy(keypoints)
    keypoint_indices = np.arange(len(keypoints))

    # Calculate average pose position
    average_landmark_pos = np.mean(keypoints, axis=0)

    # Calculate distances from average position
    distances = []
    landmark_distances = []
    for frame in keypoints:
        landmark_distances_ = np.sqrt(np.sum(np.square(frame - average_landmark_pos), axis=0))
        landmark_distances.append(landmark_distances_)
        distances.append(np.sum(landmark_distances_))
    distances = np.array(distances)

    # Smooth the distance signal
    window_size = 10
    distances = np.convolve(distances, np.ones(window_size) / window_size, mode='valid')
    keypoint_indices = keypoint_indices[window_size // 2:-window_size // 2 + 1]

    # Identify relaxed and engaged positions
    relaxed_val = min(distances)
    relaxed_threshold = 1.15 * relaxed_val
    relaxed_points = [i for i in range(len(distances)) if distances[i] < relaxed_val + relaxed_threshold]

    engaged_val = max(distances[relaxed_points[0]:relaxed_points[-1]])
    engaged_threshold = 0.30 * (engaged_val - relaxed_val)
    engaged_points = [i for i in range(len(distances)) if engaged_val - engaged_threshold < distances[i]]

    # Find rep split points
    split_points = []  # Points where motion goes: relaxed -> engaged -> relaxed
    for i in range(len(relaxed_points) - 1):
        next_point = relaxed_points[i + 1]
        current_point = relaxed_points[i]
        if next_point - current_point > 1:
            if len([j for j in engaged_points if current_point < j < next_point]) > 0:
                split_points.append(relaxed_points[i + 3])

    # Extract individual reps
    reps = [keypoints[keypoint_indices[relaxed_points[0]:split_points[0]]]]
    for i in range(len(split_points) - 1):
        rep = keypoint_indices[split_points[i]:split_points[i + 1]]
        reps.append(keypoints[rep])

    # Standardize rep lengths and average them
    rep_len = min(len(rep) for rep in reps)
    reps = [rep[len(rep) - rep_len:] for rep in reps]
    average_rep = np.mean(reps, axis=0)

    # Generate evenly spaced samples
    num_samples = 40
    reference_poses = np.array([
        average_rep[int((len(average_rep) - 1) * (i / num_samples))]
        for i in range(num_samples + 1)
    ])

    # Save reference data
    np.save(f"exercises/{exercise_name}/keyframes", reference_poses, allow_pickle=False)

    # Save regions of interest
    roi_keypoints = set()
    model_regions = get_body_segments()
    for region in regions_of_interest:
        for keypoint_id in model_regions[region]:
            roi_keypoints.add(keypoint_id)
    np.save(f"exercises/{exercise_name}/roi", np.array(list(roi_keypoints), dtype=int), allow_pickle=False)

    # Visualize reference poses
    animate_keypoint_sequence(reference_poses, exercise_name, np.arange(num_samples))

    video_capture.release()
    cv2.destroyAllWindows()


def count_reps_live(reference_exercise: ReferenceExercise, target_reps: int, form_strictness: float) -> None:
    """
    Counts exercise repetitions in real-time using webcam input.

    Args:
        reference_exercise: Reference exercise to compare against
        target_reps: Number of repetitions to complete
        form_strictness: Threshold for form accuracy (0-100)
    """
    completed_reps = 0
    keypoints = []
    rep_completions = []

    camera, video_writer = initialize_camera("to_be_deleted")
    pose_landmarker, model_options = initialize_pose_model()

    with pose_landmarker.create_from_options(model_options) as landmarker:
        reset_pos = None
        session_complete = False

        while not session_complete:
            ret, frame = camera.read()
            frame_time = int(camera.get(cv2.CAP_PROP_POS_MSEC))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker_result = landmarker.detect_for_video(mp_image, frame_time)

            annotated_frame = visualize_pose_landmarks(frame, landmarker_result)
            cv2.imshow('Camera', annotated_frame)

            if len(landmarker_result.pose_world_landmarks) > 0:
                current_pose = convert_landmarks_to_numpy([landmarker_result.pose_world_landmarks[0]])[0]
                keypoints.append(current_pose)

                ref_poses = reference_exercise.keypoint_keyframes[
                            reference_exercise.relaxed_position_index:reference_exercise.engaged_position_index
                            ]

                # Calculate distances to reference poses
                distances = [
                    np.sum(calculate_pose_distance(ref, current_pose, reference_exercise.regions_of_interest))
                    for ref in ref_poses
                ]
                rep_completion = (np.argmin(distances) / len(distances)) * 100
                rep_completions.append(rep_completion)

                # Check rep completion
                if rep_completion < (100 - form_strictness) and reset_pos == False:
                    reset_pos = True
                    print("rep complete")
                elif rep_completion > form_strictness and reset_pos == True:
                    reset_pos = False
                    completed_reps += 1
                    print(completed_reps)
                # none doesnt eval to == true or == false
                elif rep_completion < (100 - form_strictness) and reset_pos != True and reset_pos != False:
                    reset_pos = True
                    print("starting exercise")

                if completed_reps >= target_reps and reset_pos:
                    session_complete = True

            if cv2.waitKey(1) == ord('q'):
                break

    np.save("results/to_be_deleted.npy", keypoints)
    camera.release()
    video_writer.release()
    cv2.destroyAllWindows()

    plt.plot(rep_completions)
    plt.show()


def compare_to_reference(exercise_name: str, ref_exercise: ReferenceExercise) -> None:
    """
    Compares recorded exercise performance against reference exercise for form analysis.

    Args:
        exercise_name: Name of the exercise being compared
        ref_exercise: Reference exercise data to compare against
    """
    form_score = 0
    keypoint_frames = np.load("results/to_be_deleted.npy")

    # Prepare poses for visualization
    keypoint_frames = reorient_pose_data(keypoint_frames)
    reference_frames = reorient_pose_data(ref_exercise.keypoint_keyframes)
    keypoint_connections = get_pose_connections()

    # Set up 3D visualization
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(keypoint_frames[0, 0], keypoint_frames[0, 1], keypoint_frames[0, 2])

    def update_point(n):
        plt.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # Find closest reference pose
        ref_poses = reference_frames[ref_exercise.relaxed_position_index:ref_exercise.engaged_position_index]
        distances = [
            np.sum(calculate_pose_distance(ref, keypoint_frames[n], ref_exercise.regions_of_interest))
            for ref in ref_poses
        ]
        closest_ref = ref_poses[np.argmin(distances)]

        # Align poses
        left_hip_rotation = calculate_rotation_matrix(
            keypoint_frames[n, :, 23], closest_ref[:, 23]
        )
        right_hip_rotation = calculate_rotation_matrix(
            keypoint_frames[n, :, 24], closest_ref[:, 24]
        )
        avg_rotation = (left_hip_rotation + right_hip_rotation) / 2
        aligned_keypoints = avg_rotation.dot(keypoint_frames[n])

        # Update form score
        distances = [
            np.sum(calculate_pose_distance(ref, aligned_keypoints, ref_exercise.regions_of_interest))
            for ref in ref_poses
        ]
        closest_ref = ref_poses[np.argmin(distances)]
        nonlocal form_score
        form_score += np.min(distances)

        # Plot user keypoints
        ax.scatter(aligned_keypoints[0], aligned_keypoints[1], aligned_keypoints[2],
                   color='blue', label='user')
        for i in range(len(keypoint_connections)):
            for j in keypoint_connections[i]:
                if j > i:
                    ax.plot([aligned_keypoints[0, i], aligned_keypoints[0, j]],
                            [aligned_keypoints[1, i], aligned_keypoints[1, j]],
                            [aligned_keypoints[2, i], aligned_keypoints[2, j]],
                            color='blue')

        # Plot reference keypoints
        ax.scatter(closest_ref[0], closest_ref[1], closest_ref[2],
                   color='black', label='reference')
        for i in range(len(keypoint_connections)):
            for j in keypoint_connections[i]:
                if j > i:
                    ax.plot([closest_ref[0, i], closest_ref[0, j]],
                            [closest_ref[1, i], closest_ref[1, j]],
                            [closest_ref[2, i], closest_ref[2, j]],
                            color='black')

        # Plot differences between keypoints
        for i in ref_exercise.regions_of_interest:
            ax.plot([closest_ref[0, i], aligned_keypoints[0, i]],
                    [closest_ref[1, i], aligned_keypoints[1, i]],
                    [closest_ref[2, i], aligned_keypoints[2, i]],
                    color='orange')

    ani = animation.FuncAnimation(fig, update_point, frames=len(keypoint_frames) - 1)
    writer = animation.FFMpegWriter(fps=5, extra_args=['-vcodec', 'libx264'])
    ani.save(f'results/{exercise_name}_3d_w_ref.mp4', writer=writer)

    print(f"Form score: {form_score / len(keypoint_frames):.2f}")



