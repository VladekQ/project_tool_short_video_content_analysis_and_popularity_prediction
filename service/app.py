import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tempfile
import pickle
from datetime import datetime
from transformers import pipeline
from ultralytics import YOLO, YOLOWorld
from features_creation import extract_all_featues
import pandas as pd
import uuid
import joblib
np.random.seed(42)
# os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
# os.environ["NUMBA_DISABLE_JIT"] = "1"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
# model_pkl_file = r'..\machine_learning\regression_LGBM_model.pkl'
model_pkl_file = r'D:\Studying\2nd Course 2024 and 2025\project_odsv_v2\machine_learning\regression_LGBM_model.pkl'
face_recog_model = YOLO(r'D:\Studying\2nd Course 2024 and 2025\project_odsv_v2\features_extracting\object_detection\face_recognition_adam_codd_model.pt')
mwk_recog_model = YOLOWorld(r'D:\Studying\2nd Course 2024 and 2025\project_odsv_v2\features_extracting\object_detection\yolov8x-worldv2.pt')
mwk_recog_model.set_classes(['man', 'woman', 'kid'])
text_recog_model = YOLO(r'D:\Studying\2nd Course 2024 and 2025\project_odsv_v2\features_extracting\object_detection\text_recognition\training_1\weights\best.pt')
coco_recog_model = YOLO(r'D:\Studying\2nd Course 2024 and 2025\project_odsv_v2\features_extracting\object_detection\yolo11s.pt')
scaler = joblib.load(r'D:\Studying\2nd Course 2024 and 2025\project_odsv_v2\machine_learning\StandardScaler_.save') 

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Model Loading
try:
    with open(model_pkl_file, 'rb') as file:  
        model = pickle.load(file)
except:
    raise ValueError('Model is not loaded')

model_name = 'MarekCech/GenreVim-Music-Detection-DistilHuBERT'
pipe = pipeline('audio-classification', model=model_name, device=0, batch_size=18)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video(video_path, parameters):
    """Функция для обработки видео и получения прогноза"""
    # Здесь должна быть ваша реальная логика обработки
    # Для примера: просто возвращаем параметры и информацию о видео
    print(video_path)
    themes_dict = {'Singing and dancing': 119.0,
    'Humor': 104.0,
    'Sport': 112.0,
    'Anime and comics': 100.0,
    'Relationship': 107.0,
    'Show': 101.0,
    'LipSync': 110.0,
    'Everyday life': 105.0,
    'Beauty and care': 102.0,
    'Games': 103.0,
    'Society': 114.0,
    'Fashionable looks': 109.0,
    'Cars': 115.0,
    'Food': 111.0,
    'Animals': 113.0,
    'Family': 106.0,
    'Drama': 108.0,
    'Fitness and health': 117.0,
    'Education': 116.0,
    'Technologies': 118.0}

    # 7474400317610872110
    # Получаем информацию о видео
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    file_size_mb = os.path.getsize(video_path) / 1024
    video_bitrate = file_size_mb / ((duration / 60) * 0.0075)
    video_categoryType = themes_dict[parameters['video_topic']]

    top_hashtags = ['fyp', 'viral', 'foryou', 'foryoupage', 'fypシ', 'funny', 'trending',
       'tiktok', 'asmr', 'viralvideo', 'fypシ゚viral', 'cute', 'parati', 'usa',
       'cat', 'funnyvideos', 'love', 'relatable', 'fy',
       'fyppppppppppppppppppppppp', 'catsoftiktok', 'prank', 'dogsoftiktok',
       'fypage', 'baby', 'animation', 'trend', 'dog', 'couple', 'satisfying',
       'reflexion', 'challenge', 'gym', 'creatorsearchinsights', 'ai', 'car',
       'prom', 'drama', 'history', 'nyc', 'explore',
       'paratiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii', 'mukbang', 'cars',
       'unitedstates', 'usa🇺🇸', 'pet', 'edit', 'singing', 'viraltiktok']
    top_hashtags = ['#'+x for x in top_hashtags]
    df_hashtags = pd.DataFrame(index=[video_path])
    
    def check_hashtag(hashtags_str, hashtag):
        if pd.isna(hashtags_str):
            return 0
        return 1 if hashtag in hashtags_str.split() else 0

    for hashtag in top_hashtags:
        df_hashtags['hashtag_{}'.format(hashtag.replace('#', ''))] = check_hashtag(parameters['hashtags'], hashtag)
    df_hashtags.index = [video_path]
    # print(df_hashtags)
    desc_length = len(parameters['description'])
    has_question_mark = pd.Series(parameters['description']).str.contains(r'\?', regex=True).astype(int).values[0]
    has_mention = pd.Series(parameters['description']).str.contains(r'\@', regex=True).astype(int).values[0]

    df_themes = pd.DataFrame(columns=sorted(['video_theme_' + x for x in themes_dict.keys()]), index=[video_path])
    df_themes = df_themes.fillna(0)
    df_themes['video_theme_' + parameters['video_topic']] = 1

    publish_date = f"{parameters['publish_date']} {parameters['publish_time']}"
    features, music = extract_all_featues(video_path, publish_date, pipe, face_recog_model, mwk_recog_model, text_recog_model, coco_recog_model)
    # print(features)

    final_df = pd.DataFrame({
        'author_verified': [parameters['author_verified']],
        'author_followerCount': [parameters['author_followers']],
        'author_followingCount': [parameters['author_following']],
        'author_videoCount': [parameters['author_videos']],
        'author_heartCount': [parameters['author_hearts']],
        'author_diggCount': [parameters['author_digg']],
        'video_duration': [duration],
        'video_categoryType': [video_categoryType],
        'video_subtitles_languages': [parameters['subtitle_languages']],
        'video_isAd': [parameters['video_isAd']],
        'bitrate': [video_bitrate],
        'music_duration': [duration],
        'music_isCopyrighted': [parameters['music_copyrighted']],
        'music_original': [parameters['music_original']],
        'music_applemusic': [parameters['music_on_apple']]
    }, index=[video_path])

    final_df = final_df.join(features)
    final_df = final_df.join(df_hashtags)

    desc_df = pd.DataFrame({
        'desc_length': [desc_length],
        'has_question_mark': [has_question_mark],
        'has_mention': [has_mention]
    }, index=[video_path])

    final_df = final_df.join(desc_df)
    final_df = final_df.join(df_themes)
    final_df = final_df.join(music)
    print(final_df.shape)
    final_df = final_df.drop('fps', axis=1)
    # print(file_size_mb)
    # Формируем результат
    final_df.to_excel('example.xlsx')
    final_df_scaled = scaler.transform(final_df)
    
    prediction = model.predict(final_df_scaled)
    print(2**prediction)
    return {
        'estimated_views': int(2**prediction[0]),
        'parameters': parameters
    }
    # result = {
    #     'video_info': {
    #         'duration': round(duration, 2),
    #         'fps': round(fps, 2),
    #         'frame_count': frame_count,
    #         'filename': os.path.basename(video_path)
    #     },
    #     'parameters': parameters,
    #     'file_size': file_size_mb,
    #     'prediction': {
    #         'Estimated number of views for the video': 2**prediction[0]
    #     }
    #     # 'prediction': {
    #     #     'success_rate': 0.85,  # Пример предсказания
    #     #     'estimated_views': 10000 * parameters['author_followers'] / 1000 if parameters['author_followers'] > 0 else 5000,
    #     #     'recommendations': [
    #     #         "Add more hashtags" if len(parameters['hashtags'].split()) < 3 else "Hashtags count is good",
    #     #         "Consider adding subtitles" if not parameters['has_subtitles'] else "Subtitles will help",
    #     #         f"Topic '{parameters['video_topic']}' is trending" if parameters['video_topic'] in ['Humor', 'Animals'] else f"Topic '{parameters['video_topic']}' is stable"
    #     #     ]
    #     # }
    # }
    
    return result

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # filepath = os.path.abspath(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # temp_filename = f"temp_{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}"
            # filepath = os.path.join(os.getcwd(), temp_filename)
            # file.save(filepath)
            
            # Получаем все параметры из формы
            parameters = {
                # Author Parameters
                'author_verified': request.form.get('author_verified') == 'on',
                'author_followers': int(request.form.get('author_followers', 0)),
                'author_following': int(request.form.get('author_following', 0)),
                'author_videos': int(request.form.get('author_videos', 0)),
                'author_hearts': int(request.form.get('author_hearts', 0)),
                'author_digg': int(request.form.get('author_digg', 0)),
                
                # Video Features
                'has_subtitles': request.form.get('has_subtitles') == 'on',
                'subtitle_languages': int(request.form.get('subtitle_languages', 0)),
                'hashtags': request.form.get('hashtags', ''),
                'description': request.form.get('description', ''),
                'video_topic': request.form.get('video_topic', ''),
                'video_isAd': request.form.get('is_ad', 0),
                
                # Music Features
                'has_music': request.form.get('has_music') == 'on',
                'music_copyrighted': request.form.get('music_copyrighted') == 'on',
                'music_original': request.form.get('music_original') == 'on',
                'music_on_apple': request.form.get('music_on_apple') == 'on',

                # Publication DateTime
                'publish_date': request.form.get('publish_date'),
                'publish_time': request.form.get('publish_time'),
                'publish_datetime': f"{request.form.get('publish_date')}T{request.form.get('publish_time')}",
            }
            
            # Обрабатываем видео
            result = process_video(filepath, parameters)
            
            # Удаляем временный файл
            os.remove(filepath)
            # try:
            #     os.remove(filepath)
            # except Exception as e:
            #     print(f"Error deleting temp file: {e}")
            
            return render_template('result.html', estimated_views=result['estimated_views'], parameters=result['parameters'])# result=result)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)