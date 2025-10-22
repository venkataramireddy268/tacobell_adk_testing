import json, cv2, os

def extract_keyframes(
    video_path="/home/nagababu_upputuri/cloudshell_open/Naga_GenAI/ai_generated_videos_Taco_bell_ad.mp4", 
    json_path="/home/nagababu_upputuri/cloudshell_open/Naga_GenAI/output.json", 
    output_dir="keyframes"
    ):
    # load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for scene in data:
        # use the start timestamp to grab the representative frame
        timestamp = scene["start_timestamp"]
        h, m, s = map(float, timestamp.split(":"))
        frame_no = int((h*3600 + m*60 + s) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame for scene {scene['scene_number']}")
            continue

        out_file = os.path.join(output_dir, scene["keyframe_image"])
        cv2.imwrite(out_file, frame)
        print(f"Saved keyframe: {out_file}")

    cap.release()
    print("All keyframes extracted successfully!")
extract_keyframes()