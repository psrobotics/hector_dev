# Visualize and calibrate home qpos
import mujoco
import mujoco.viewer
import time

def visualize_home_pose(mjcf_path, key_name="home"):
    """
    Loads an MJCF model, visualizes its home pose (qpos0), 
    and freezes at that frame.

    """
    try:
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        data = mujoco.MjData(model)

        key_id = model.key(key_name).id
        print(f"Found keyframe '{key_name}' with ID: {key_id}")

        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Launch the passive viewer to display the model.
        print(f"\nDisplaying pose from keyframe '{key_name}'.")
        print("Close the viewer window to exit.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                time.sleep(0.1)


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    mjcf_file_path = "../xmls/scene_mjx_feetonly_flat_terrain.xml"
    visualize_home_pose(mjcf_file_path)