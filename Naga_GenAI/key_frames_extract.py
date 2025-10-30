import json, cv2, os

def extract_keyframes(
    # video_path="/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/Baaki se breakup, New Cheesy Lava Taco ke saath karo makeup..mp4",
    video_path="/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/cloudshell_open/tacobell_adk_testing/Naga_GenAI/cloudshell_open/tacobell_adk_testing/Naga_GenAI/official_8s_video.mp4",
    json_path="/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/cloths_loc1.json",
    output_dir="keyframes_v2_1"
    ):
    # load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for scene in data:
        # use the start timestamp to grab the representative frame
        timestamp = scene["start timestamp"]
        h, m, s = map(float, timestamp.split(":"))
        frame_no = int((h*3600 + m*60 + s) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame for scene {scene['Scene Number']}")
            continue

        out_file = os.path.join(output_dir, f'scene_{scene["Scene Number"]}.jpg')
        cv2.imwrite(out_file, frame)
        print(f"Saved keyframe: {out_file}")

    cap.release()
    print("All keyframes extracted successfully!")
# extract_keyframes()

import json, cv2, os

def extract_keyframes(
    video_path="/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/official_8s_video.mp4",
    json_path="/home/nagababu_upputuri/cloudshell_open/tacobell_adk_testing/Naga_GenAI/cloths_loc_1.json",
    output_dir="keyframes_single_1"
):
    # Load scene JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Video FPS: {fps}")

    # Helper: convert timestamp → seconds
    def ts_to_seconds(ts):
        h, m, s = map(float, ts.split(":"))
        return h * 3600 + m * 60 + s

    for scene in data:
        scene_num = scene["Scene Number"]
        start_ts = scene["start timestamp"]
        end_ts = scene["end timestamp"]

        # Take middle of the scene duration (better representative frame)
        start_sec = ts_to_seconds(start_ts)
        end_sec = ts_to_seconds(end_ts)
        mid_sec = (start_sec + end_sec) / 2
        frame_no = int(mid_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ Could not read frame for scene {scene_num}")
            continue

        out_file = os.path.join(output_dir, f"scene_{scene_num}.jpg")
        cv2.imwrite(out_file, frame)
        print(f"✅ Saved keyframe: {out_file}")

    cap.release()
    print("\n🎉 All keyframes extracted successfully!")

extract_keyframes()
