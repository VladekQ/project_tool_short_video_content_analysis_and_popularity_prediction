import pandas as pd
import cv2
import numpy as np
from skimage.measure import shannon_entropy
import librosa
import subprocess
import os
from moviepy import VideoFileClip


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'

def create_date_features(date_in):
    date_in = pd.to_datetime(date_in)
    return date_in.hour, date_in.dayofweek

def extract_video_features(video_path):
    # Optimization
    FRAMES_FOR_DOMINANT_COLORS = 5
    CUT_DETECTION_THRESHOLD = 0.8
    SKIP_FRAMES = 2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Инициализация переменных
    brightness_values = []
    motion_values = []
    cut_count = 0
    prev_frame = None
    prev_hist = None
    text_presence = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % (SKIP_FRAMES + 1) != 0:
            continue  # Пропускаем кадры для ускорения
        
        # Быстрое преобразование в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Яркость (используем более быстрый метод)
        brightness = cv2.mean(gray)[0]
        brightness_values.append(brightness)
        
        # Оптический поток (только если есть предыдущий кадр)
        if prev_frame is not None:
            # Уменьшаем разрешение для ускорения расчета оптического потока
            small_prev = cv2.resize(prev_frame, None, fx=0.5, fy=0.5)
            small_curr = cv2.resize(gray, None, fx=0.5, fy=0.5)
            
            flow = cv2.calcOpticalFlowFarneback(
                small_prev, small_curr, None, 
                pyr_scale=0.5, levels=3, winsize=10, 
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )
            motion = np.mean(np.abs(flow))
            motion_values.append(motion)
        
        # Детекция склеек (упрощенный метод)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])  # Меньше бинов для скорости
        cv2.normalize(hist, hist)  # Нормализация для сравнения
        
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if diff < CUT_DETECTION_THRESHOLD:
                cut_count += 1
        
        prev_hist = hist
        prev_frame = gray
    
    cap.release()
    
    # Подготовка результата
    result = {
        "fps": fps,
        "avg_brightness": np.mean(brightness_values) if brightness_values else 0,
        "avg_motion": np.mean(motion_values) if motion_values else 0,
        "cut_rate": cut_count / duration if duration > 0 else 0,
        "entropy": shannon_entropy(gray) if 'gray' in locals() else 0,
    }
    
    return result

def extract_audio_from_video(video_path):
    try:
        # Define the input video file and output audio file
        mp4_file = video_path
        wav_file = mp4_file.replace('.mp4', '.mp3').replace('.avi', '.mp3').replace('.mov', '.mp3')
        
        # Load the video clip
        video_clip = VideoFileClip(mp4_file)
        print('Video OKAY')
        # Extract the audio from the video clip
        audio_clip = video_clip.audio
        print(wav_file)
        # Write the audio to a separate file
        audio_clip.write_audiofile(wav_file, logger=None)
        
        # Close the video and audio clips
        audio_clip.close()
        video_clip.close()
        
        return wav_file
        # print("Audio extraction successful!")
    except:
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}")

def music_non_music_audio(audio_path, pipe):
    # audio_path = extract_audio_from_video(video_path)

    audio, rate = librosa.load(audio_path)
    print('AUDIO COLLECTED')

    result = pipe(audio)[0]['label']
    return result

def extract_audio_features(audio_path):
    def calculate_snr(y, sr):
        S = librosa.stft(y)
        magnitude = np.abs(S)
        noise = np.median(magnitude)
        signal = np.max(magnitude)
        snr = 10 * np.log10(signal / noise) if noise > 0 else 100
        return snr

    def calculate_harmonicity(y):
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(harmonic**2) / (np.sum(y**2) + 1e-6)
        return harmonic_ratio

    def calculate_dynamic_range(y):
        dyn_range = np.max(y) - np.min(y)
        return dyn_range

    def calculate_spectral_centroid(y, sr):
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        return np.mean(centroid)

    def detect_clipping(y, threshold=0.99):
        clipping_samples = np.sum(np.abs(y) > threshold * np.max(np.abs(y)))
        clipping_ratio = clipping_samples / len(y)
        return clipping_ratio

    y, sr = librosa.load(audio_path)
    # duration = librosa.get_duration(y=y, sr=sr)
    
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_avg = np.mean(mfcc, axis=1)
    
    rms = librosa.feature.rms(y=y).mean()
    
    chunk_duration = 5
    chunk_samples = int(chunk_duration * sr)
    chunks = [y[i:i + chunk_samples] for i in range(0, len(y), chunk_samples)]
    chunks = sorted(chunks, key=lambda x: x.shape[0], reverse=True)
    y = chunks[0]
    snr = calculate_snr(y, sr)
    harmonicity = calculate_harmonicity(y)
    dyn_range = calculate_dynamic_range(y)
    centroid = calculate_spectral_centroid(y, sr)
    clipping = detect_clipping(y)
    
    quality_score = (
        0.3 * snr +
        0.2 * harmonicity +
        0.2 * dyn_range +
        0.1 * centroid +
        0.2 * clipping
    )
    
    return {
        "tempo": tempo,
        "loudness": rms,
        **{f"mfcc_{i+1}": mfcc_avg[i] for i in range(13)},
        "quality_score": quality_score,
        "snr": snr,
        "harmonicity": harmonicity,
        "dyn_range": dyn_range,
        "centroid": centroid,
        "clipping": clipping
    }

def recognize_face(model, video_path, iou=0.1, vid_stride=10, conf=0.75):
    results = model(video_path, save=False, iou=iou, vid_stride=vid_stride, show=False, conf=conf, verbose=False)
    data = []
    for frame_idx, result in enumerate(results):
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            data.append([
                frame_idx + 1,
                result.names[int(cls)],
                float(conf),
                *box.tolist()  # x1, y1, x2, y2
            ])
            
    
    df = pd.DataFrame(data, columns=["frame", "class", "confidence", "x1", "y1", "x2", "y2"])

    df_to_append = df[df['confidence'] > 0.5].pivot_table(columns='class', values='confidence', aggfunc='sum')
    df_to_append.index = [video_path]
    # df_to_append = df_to_append.clip(upper=1)
    return df_to_append

def recognize_man_woman_kid(model, video_path, iou=0.1, vid_stride=20, conf=0.1):
    results = model(video_path, save=False, iou=iou, vid_stride=vid_stride, show=False, conf=conf, verbose=False)
    data = []
    for frame_idx, result in enumerate(results):
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            data.append([
                frame_idx + 1,
                result.names[int(cls)],
                float(conf),
                *box.tolist()  # x1, y1, x2, y2
            ])
            
    
    df = pd.DataFrame(data, columns=["frame", "class", "confidence", "x1", "y1", "x2", "y2"])
    df_to_append = df.pivot_table(columns='class', values='confidence', aggfunc='sum')
    # df_to_append
#     df_to_append = df[df['confidence'] > 0.5].pivot_table(columns='class', values='confidence', aggfunc='sum')
    df_to_append.index = [video_path]
    # df_to_append = df_to_append.clip(upper=1)
    return df_to_append

def recognize_text(model, video_path, iou=0.1, vid_stride=5, conf=0.5):
    results = model(video_path, save=False, iou=iou, vid_stride=vid_stride, show=False, conf=conf, verbose=False)
    data = []
    for frame_idx, result in enumerate(results):
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            data.append([
                frame_idx + 1,
                result.names[int(cls)],
                float(conf),
                *box.tolist()  # x1, y1, x2, y2
            ])
            
    
    df = pd.DataFrame(data, columns=["frame", "class", "confidence", "x1", "y1", "x2", "y2"])
    df_to_append = df.pivot_table(columns='class', values='confidence', aggfunc='sum')
    # df_to_append = df[df['confidence'] > 0.5].pivot_table(columns='class', values='confidence', aggfunc='sum')
    df_to_append.index = [video_path]
    # df_to_append = df_to_append.clip(upper=1)
    return df_to_append

def recognize_coco(model, video_path, iou=0.1, vid_stride=5, conf=0.7):
    results = model(video_path, save=False, iou=iou, vid_stride=vid_stride, show=False, conf=conf, verbose=False)
    data = []
    for frame_idx, result in enumerate(results):
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            data.append([
                frame_idx + 1,
                result.names[int(cls)],
                float(conf),
                *box.tolist()  # x1, y1, x2, y2
            ])
            
    
    df = pd.DataFrame(data, columns=["frame", "class", "confidence", "x1", "y1", "x2", "y2"])
    df_to_append = df.pivot_table(columns='class', values='confidence', aggfunc='sum')
    # df_to_append = df[df['confidence'] > 0.5].pivot_table(columns='class', values='confidence', aggfunc='sum')
    df_to_append.index = [video_path]
    # df_to_append = df_to_append.clip(upper=1)
    return df_to_append

def extract_all_featues(video_path, publish_date, pipe, face_recog_model, mwk_recog_model, text_recog_model, coco_recog_model):
    print('Calculating Date Features')
    createTime_hour, createTime_weekday = create_date_features(publish_date)
    print('Calculating Video Features')
    video_features = extract_video_features(video_path)
    print('Extracting Audio')
    audio_path = extract_audio_from_video(video_path)
    print('Calculating Music Features')
    music_non_music = music_non_music_audio(audio_path, pipe)
    print('Calculating Audio Features')
    audio_features = extract_audio_features(audio_path)
    print("Recognizing Faces")
    faces = recognize_face(face_recog_model, video_path)
    if faces.shape[1] == 0:
        faces = pd.DataFrame({'face': [0]})
        faces.index = [video_path]
    else:
        faces['face'] = np.where(faces['face'] > 0, 1, 0)
    print('Recognizing Man/Woman/Kid')
    man_woman_kid = recognize_man_woman_kid(mwk_recog_model, video_path)
    man_woman_kid_columns = ['man', 'woman', 'kid']
    man_woman_kid_df = pd.DataFrame()
    for col in man_woman_kid_columns:
        try:
            man_woman_kid_df.loc[video_path, col] = man_woman_kid.loc[video_path, col]
        except:
            man_woman_kid_df.loc[video_path, col] = 0
    man_woman_kid_df[man_woman_kid_df.columns] = np.where(man_woman_kid_df > 1, 1, 0)
    # if man_woman_kid.shape[1] == 0:
    #     man_woman_kid = pd.DataFrame({'man': [0], 'woman': [0], 'kid': [0]})
    #     man_woman_kid.index = [video_path]
    # else:
    #     man_woman_kid[man_woman_kid.columns] = np.where(man_woman_kid > 1, 1, 0)
    #     man_woman_kid = man_woman_kid[['man', 'woman', 'kid']]
    print('Recognizing Text')
    texts = recognize_text(text_recog_model, video_path)
    if texts.shape[1] == 0:
        texts = pd.DataFrame({'text_on_picutre': [0]})
        texts.index = [video_path]
    else:
        texts[texts.columns] = np.where(texts > 1, 1, 0)
    print("Recognizing COCO")
    coco = recognize_coco(coco_recog_model, video_path)

    # coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    # 'fire hydrant',  'stop sign',  'parking meter',  'bench',  
    # 'bird',  'cat',  'dog',  'horse',  'sheep',  'cow',  'elephant',  
    # 'bear',  'zebra',  'giraffe',  'backpack',  'umbrella',  'handbag',  
    # 'tie',  'suitcase',  'frisbee',  'skis',  'snowboard',  'sports ball', 
    # 'kite',  'baseball bat',  'baseball glove',  'skateboard',  'surfboard',  
    # 'tennis racket',  'bottle',  'wine glass',  'cup',  'fork',  'knife',  'spoon',  
    # 'bowl',  'banana',  'apple',  'sandwich',  'orange',  'broccoli',  'carrot',  
    # 'hot dog',  'pizza',  'donut',  'cake',  'chair',  'couch',  'potted plant',  
    # 'bed',  'dining table',  'toilet',  'tv',  'laptop',  'mouse',  'remote',  
    # 'keyboard',  'cell phone',  'microwave',  'oven',  'toaster',  'sink',  
    # 'refrigerator',  'book',  'clock',  'vase',  'scissors',  'teddy bear',  
    # 'hair drier',  'toothbrush']
    coco_classes = ['person', 'bottle', 'bowl', 'cup', 'toilet', 'toothbrush', 'kite',
    'umbrella', 'car', 'cell phone', 'truck', 'handbag', 'hot dog', 
    'remote', 'tie', 'tv', 'cat', 'couch', 'dog', 'horse', 'refrigerator', 
    'banana', 'motorcycle', 'skateboard', 'bed', 'oven', 'dining table', 
    'teddy bear', 'clock', 'knife', 'pizza', 'airplane', 'laptop', 'sandwich', 
    'train', 'frisbee', 'sports ball', 'book', 'mouse', 'spoon', 'vase', 'bird', 
    'sink', 'traffic light', 'apple', 'scissors', 'bus', 'chair', 'bicycle', 
    'potted plant', 'cake', 'cow', 'surfboard', 'backpack', 'tennis racket',
    'baseball bat', 'donut', 'boat', 'wine glass', 'broccoli', 'bench', 
    'suitcase', 'orange', 'keyboard', 'fire hydrant', 'sheep', 'fork', 
    'giraffe', 'carrot', 'microwave', 'zebra', 'elephant', 'bear', 
    'baseball glove', 'hair drier', 'parking meter', 'stop sign', 
    'skis', 'snowboard', 'toaster']
    coco_df = pd.DataFrame()
    for col in coco_classes:
        try:
            coco_df.loc[video_path, col] = coco.loc[video_path, col]
        except:
            coco_df.loc[video_path, col] = 0
    coco_df[coco_df.columns] = np.where(coco_df > 0.8, 1, 0)
    # if coco.shape[1] == 0:
    #     coco = pd.DataFrame(columns=coco_classes, index=[video_path])
    #     coco = coco.fillna(0)
    # else:
    #     coco[coco.columns] = np.where(coco > 1, 1, 0)

    final_df = faces.join(man_woman_kid_df).join(texts).join(coco_df)
    video_features = pd.DataFrame.from_dict(video_features, orient='index').T
    video_features.index = [video_path]
    audio_features = pd.DataFrame.from_dict(audio_features, orient='index').T
    audio_features.index = [video_path]

    music = pd.DataFrame(columns=['music_non_music_Music', 'music_non_music_Non Music'], index=[video_path]).fillna(0)
    if music_non_music == 'Music':
        music['music_non_music_Music'] = 1
    else:
        music['music_non_music_Non Music'] = 1

    final_df = final_df.join(video_features).join(audio_features)
    return final_df, music
    # return createTime_hour, createTime_weekday, video_features, music_non_music, audio_features, faces, man_woman_kid, texts, coco
