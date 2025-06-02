import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from dash.dash_table import DataTable

# Подключение к БД
conn = sqlite3.connect('videos_database.db', check_same_thread=False)

app = dash.Dash(__name__)

# Настройки шрифта
app.layout = html.Div(style={'font-family': 'Roboto, Arial, sans-serif'}, children=[
    html.H1("Video Content Analytics Dashboard", style={'textAlign': 'center', 'font-family': 'Roboto'}),
    
    # Фильтры
    html.Div([
        html.Div([
            html.Label("Date range:"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=datetime(2023, 1, 1),
                max_date_allowed=datetime.today(),
                start_date=datetime.today() - timedelta(days=30),
                end_date=datetime.today()
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Video theme:"),
            dcc.Dropdown(
                id='theme-filter',
                options=[{'label': 'All', 'value': 'all'}] + 
                       [{'label': theme, 'value': theme} 
                        for theme in pd.read_sql("SELECT DISTINCT video_theme FROM videos_metadata_to_analyze", conn)['video_theme']],
                value='all',
                multi=True
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Minimum views:"),
            dcc.Input(
                id='views-filter',
                type='number',
                value=0,
                min=0
            )
        ], style={'width': '20%', 'display': 'inline-block'})
    ], style={'margin-bottom': '20px'}),
    
    # KPI метрики
    html.Div(id='kpi-metrics', style={
        'display': 'flex',
        'justify-content': 'space-between',
        'margin-bottom': '20px'
    }),
    
    # Основные графики
    dcc.Graph(id='avg-views-by-theme', style={'width': '100%', 'height': '400px'}),
    dcc.Graph(id='videos-count-by-theme', style={'width': '100%', 'height': '400px'}),
    dcc.Graph(id='views-trend', style={'width': '100%', 'height': '400px'}),
    dcc.Graph(id='avg-views-trend-by-theme', style={'width': '100%', 'height': '500px'}),
    dcc.Graph(id='duration-distribution', style={'width': '100%', 'height': '400px'}),
    
    # Топ-таблицы
    html.Div([
        html.Div([
            html.H3("Top 10 Videos"),
            html.Div(id='top-videos-table')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Top 10 Authors"),
            html.Div(id='top-authors-table')
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ], style={'margin-top': '20px'}),
    
    # Анализ хештегов
    dcc.Graph(id='hashtag-chart', style={'width': '100%', 'height': '500px', 'margin-top': '20px'}),

    html.H2("Object Analysis", style={'margin-top': '40px'}),
    
    html.Div([
        html.Div([
            dcc.Graph(id='top-objects', style={'width': '100%', 'height': '500px'})
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='views-by-objects', style={'width': '100%', 'height': '500px'})
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ])
])

# Общие стили для графиков
common_graph_style = {
    'margin': '20px 0',
    'border': '1px solid #eee',
    'border-radius': '8px',
    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
}

# Callback для KPI метрик
@app.callback(
    Output('kpi-metrics', 'children'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-filter', 'value'),
     Input('views-filter', 'value')]
)
def update_kpi(start_date, end_date, themes, min_views):
    query = """
    SELECT 
        COUNT(DISTINCT video_id) as total_videos,
        SUM(playCount) as total_views,
        AVG(playCount) as avg_views,
        COUNT(DISTINCT author_nickname) as unique_authors
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    AND playCount >= ?
    """
    
    params = [start_date, end_date, min_views]
    
    if themes != 'all' and themes:
        if isinstance(themes, str):
            themes = [themes]
        query += " AND video_theme IN ({})".format(','.join(['?']*len(themes)))
        params.extend(themes)
    
    kpi = pd.read_sql(query, conn, params=params).iloc[0]
    
    return [
        html.Div([
            html.H3(f"{kpi['total_videos']:,}"),
            html.P("Total Videos")
        ], style={'text-align': 'center', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '8px'}),
        
        html.Div([
            html.H3(f"{kpi['total_views']:,}"),
            html.P("Total Views")
        ], style={'text-align': 'center', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '8px'}),
        
        html.Div([
            html.H3(f"{kpi['avg_views']:,.0f}"),
            html.P("Average Views")
        ], style={'text-align': 'center', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '8px'}),
        
        html.Div([
            html.H3(f"{kpi['unique_authors']:,}"),
            html.P("Unique Authors")
        ], style={'text-align': 'center', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '8px'})
    ]

# Callback для графика по темам
@app.callback(
    Output('avg-views-by-theme', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('views-filter', 'value')]
)
def update_avg_views_by_theme(start_date, end_date, min_views):
    query = """
    SELECT 
        video_theme,
        AVG(playCount) as avg_views,
        COUNT(video_id) as videos_count
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    AND playCount >= ?
    GROUP BY video_theme
    --HAVING COUNT(video_id) >= 5
    ORDER BY avg_views DESC
    LIMIT 15
    """
    
    df = pd.read_sql(query, conn, params=[start_date, end_date, min_views])
    
    fig = px.bar(
        df, 
        x='video_theme', 
        y='avg_views',
        title='Average Views by Video Theme (min 5 videos per theme)',
        labels={'video_theme': 'Video Theme', 'avg_views': 'Average Views'},
        hover_data=['videos_count'],
        color='avg_views',
        color_continuous_scale='Bluered'
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Roboto'},
        yaxis_title="Average Views"
    )
    return fig

# Callback для количества видео по темам
@app.callback(
    Output('videos-count-by-theme', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('views-filter', 'value')]
)
def update_videos_count_by_theme(start_date, end_date, min_views):
    query = """
    SELECT 
        video_theme,
        COUNT(video_id) as videos_count,
        AVG(playCount) as avg_views
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    AND playCount >= ?
    GROUP BY video_theme
    ORDER BY videos_count DESC
    LIMIT 15
    """
    
    df = pd.read_sql(query, conn, params=[start_date, end_date, min_views])
    
    fig = px.bar(
        df, 
        x='video_theme', 
        y='videos_count',
        title='Number of Videos by Theme',
        labels={'video_theme': 'Video Theme', 'videos_count': 'Number of Videos'},
        hover_data=['avg_views'],
        color='videos_count',
        color_continuous_scale='Teal'
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Roboto'},
        yaxis_title="Number of Videos",
        coloraxis_colorbar=dict(title='Count')
    )
    return fig

# Callback для временного тренда
@app.callback(
    Output('views-trend', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-filter', 'value'),
     Input('views-filter', 'value')]
)
def update_trend_chart(start_date, end_date, themes, min_views):
    query = """
    SELECT 
        date(createTime) as day,
        SUM(playCount) as total_views,
        COUNT(video_id) as videos_count
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    AND playCount >= ?
    """
    
    params = [start_date, end_date, min_views]
    
    if themes != 'all' and themes:
        if isinstance(themes, str):
            themes = [themes]
        query += " AND video_theme IN ({})".format(','.join(['?']*len(themes)))
        params.extend(themes)
    
    query += " GROUP BY day ORDER BY day"
    
    df = pd.read_sql(query, conn, params=params)
    
    fig = px.area(
        df, 
        x='day', 
        y='total_views',
        title='Views Trend Over Time',
        labels={'day': 'Date', 'total_views': 'Views'},
        hover_data=['videos_count']
    )
    
    fig.add_bar(
        x=df['day'],
        y=df['videos_count'],
        name='Videos Count',
        yaxis='y2'
    )
    
    fig.update_layout(
        yaxis2=dict(
            title='Videos Count',
            overlaying='y',
            side='right'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Roboto'}
    )
    
    return fig

# Callback для тренда по темам
@app.callback(
    Output('avg-views-trend-by-theme', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-filter', 'value'),
     Input('views-filter', 'value')]
)
def update_avg_views_trend_by_theme(start_date, end_date, themes, min_views):
    query = """
    SELECT 
        date(createTime) as day,
        video_theme,
        AVG(playCount) as avg_views,
        COUNT(video_id) as videos_count
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    AND playCount >= ?
    """
    
    params = [start_date, end_date, min_views]
    
    if themes != 'all' and themes:
        if isinstance(themes, str):
            themes = [themes]
        query += " AND video_theme IN ({})".format(','.join(['?']*len(themes)))
        params.extend(themes)
    
    query += """
    GROUP BY day, video_theme
    HAVING COUNT(video_id) >= 3
    ORDER BY day, video_theme
    """
    
    df = pd.read_sql(query, conn, params=params)
    
    # Оставляем только топ-5 тем по средним просмотрам
    top_themes = df.groupby('video_theme')['avg_views'].mean().nlargest(5).index
    df = df[df['video_theme'].isin(top_themes)]
    
    fig = px.line(
        df, 
        x='day', 
        y='avg_views',
        color='video_theme',
        title='Daily Average Views by Top 5 Themes (min 3 videos per day)',
        labels={'day': 'Date', 'avg_views': 'Average Views', 'video_theme': 'Theme'},
        hover_data=['videos_count'],
        line_shape='spline',
        render_mode='svg'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Roboto'},
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title="Date",
        yaxis_title="Average Views"
    )
    
    # Добавляем маркеры для точек данных
    fig.update_traces(mode='lines+markers', marker=dict(size=6))
    
    return fig

# Callback для распределения длительности
@app.callback(
    Output('duration-distribution', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-filter', 'value'),
     Input('views-filter', 'value')]
)
def update_duration_chart(start_date, end_date, themes, min_views):
    query = """
    SELECT 
        video_duration,
        playCount
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    AND playCount >= ?
    """
    
    params = [start_date, end_date, min_views]
    
    if themes != 'all' and themes:
        if isinstance(themes, str):
            themes = [themes]
        query += " AND video_theme IN ({})".format(','.join(['?']*len(themes)))
        params.extend(themes)
    
    df = pd.read_sql(query, conn, params=params)
    
    fig = px.scatter(
        df, 
        x='video_duration', 
        y='playCount',
        title='Views by Video Duration',
        labels={'video_duration': 'Duration (seconds)', 'playCount': 'Views'},
        trendline="lowess"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Roboto'}
    )
    
    return fig

# Callback для топ-видео (теперь топ-20)
@app.callback(
    Output('top-videos-table', 'children'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-filter', 'value'),
     Input('views-filter', 'value')]
)
def update_top_videos(start_date, end_date, themes, min_views):
    query = """
    SELECT 
        video_id,
        author_nickname,
        video_theme,
        playCount as views
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    AND playCount >= ?
    """
    
    params = [start_date, end_date, min_views]
    
    if themes != 'all' and themes:
        if isinstance(themes, str):
            themes = [themes]
        query += " AND video_theme IN ({})".format(','.join(['?']*len(themes)))
        params.extend(themes)
    
    query += " ORDER BY playCount DESC LIMIT 20"  # Изменено с 10 на 20
    
    df = pd.read_sql(query, conn, params=params)
    
    return DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto', 'height': '400px', 'overflowY': 'auto'},
        style_cell={
            'fontFamily': 'Roboto',
            'textAlign': 'left',
            'padding': '8px',
            'minWidth': '100px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        page_size=10  # Пагинация по 10 строк
    )

# Callback для топ-авторов (теперь топ-20)
@app.callback(
    Output('top-authors-table', 'children'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-filter', 'value'),
     Input('views-filter', 'value')]
)
def update_top_authors(start_date, end_date, themes, min_views):
    query = """
    SELECT 
        author_nickname,
        SUM(playCount) as total_views,
        COUNT(video_id) as videos_count,
        ROUND(SUM(playCount)/COUNT(video_id), 0) as avg_views
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    AND playCount >= ?
    """
    
    params = [start_date, end_date, min_views]
    
    if themes != 'all' and themes:
        if isinstance(themes, str):
            themes = [themes]
        query += " AND video_theme IN ({})".format(','.join(['?']*len(themes)))
        params.extend(themes)
    
    query += " GROUP BY author_nickname ORDER BY total_views DESC LIMIT 20"  # Изменено с 10 на 20
    
    df = pd.read_sql(query, conn, params=params)
    
    return DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto', 'height': '400px', 'overflowY': 'auto'},
        style_cell={
            'fontFamily': 'Roboto',
            'textAlign': 'left',
            'padding': '8px',
            'minWidth': '100px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        page_size=10  # Пагинация по 10 строк
    )

# Callback для топ-10 хештегов по средним просмотрам (с фильтром по минимальному количеству)
@app.callback(
    Output('hashtag-chart', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-filter', 'value')]
)
def update_hashtag_chart(start_date, end_date, themes):
    # Сначала получаем все видео с хештегами
    query = """
    SELECT video_id, video_hashtags, playCount
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    AND video_hashtags != ''
    """
    
    params = [start_date, end_date]
    
    if themes != 'all' and themes:
        if isinstance(themes, str):
            themes = [themes]
        query += " AND video_theme IN ({})".format(','.join(['?']*len(themes)))
        params.extend(themes)
    
    df = pd.read_sql(query, conn, params=params)
    
    # Разбиваем хештеги и создаем список всех отдельных хештегов с просмотрами
    hashtags_data = []
    for _, row in df.iterrows():
        for tag in row['video_hashtags'].split():
            hashtags_data.append({
                'hashtag': tag[:].lower(),  # Убираем # и приводим к нижнему регистру
                'views': row['playCount']
            })
    
    # Создаем DataFrame
    hashtags_df = pd.DataFrame(hashtags_data)
    
    # Группируем по хештегам и считаем статистику
    hashtag_stats = hashtags_df.groupby('hashtag')['views'].agg(
        avg_views='mean',
        count='count',
        total_views='sum'
    ).reset_index()
    
    # Фильтруем хештеги с минимум 10 использованиями
    filtered_hashtags = hashtag_stats[hashtag_stats['count'] >= 10]
    
    # Сортируем по средним просмотрам и берем топ-10
    top_hashtags = filtered_hashtags.sort_values('avg_views', ascending=False).head(10)
    
    # Создаем график
    fig = px.bar(
        top_hashtags,
        x='hashtag',
        y='avg_views',
        title='Top 10 Hashtags by Average Views (min 10 uses)',
        labels={
            'hashtag': 'Hashtag', 
            'avg_views': 'Average Views',
            'count': 'Usage Count',
            'total_views': 'Total Views'
        },
        hover_data=['count', 'total_views'],
        color='avg_views',
        color_continuous_scale='Bluered'
    )
    
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Roboto'},
        hovermode='x',
        coloraxis_colorbar=dict(
            title='Avg Views'
        )
    )
    
    # Добавляем аннотацию с минимальным порогом
    fig.add_annotation(
        x=0.5,
        y=1.1,
        xref='paper',
        yref='paper',
        text="Only hashtags used 10+ times included",
        showarrow=False,
        font=dict(size=12)
    )
    
    return fig

# Callback для количества видео с объектами
@app.callback(
    Output('top-objects', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-filter', 'value')]
)
def update_top_objects(start_date, end_date, themes):
    # Список всех объектов
    objects = [
        'face', 'man', 'woman', 'kid', 'text_on_picutre', 'person', 'bottle', 'bowl', 'cup', 
        'toilet', 'toothbrush', 'kite', 'umbrella', 'car', 'cell phone', 'truck', 'handbag', 
        'hot dog', 'remote', 'tie', 'tv', 'cat', 'couch', 'dog', 'horse', 'refrigerator', 
        'banana', 'motorcycle', 'skateboard', 'bed', 'oven', 'dining table', 'teddy bear', 
        'clock', 'knife', 'pizza', 'airplane', 'laptop', 'sandwich', 'train', 'frisbee', 
        'sports ball', 'book', 'mouse', 'spoon', 'vase', 'bird', 'sink', 'traffic light', 
        'apple', 'scissors', 'bus', 'chair', 'bicycle', 'potted plant', 'cake', 'cow', 
        'surfboard', 'backpack', 'tennis racket', 'baseball bat', 'donut', 'boat', 
        'wine glass', 'broccoli', 'bench', 'suitcase', 'orange', 'keyboard', 'fire hydrant', 
        'sheep', 'fork', 'giraffe', 'carrot', 'microwave', 'zebra', 'elephant', 'bear', 
        'baseball glove', 'hair drier', 'parking meter', 'stop sign', 'skis', 'snowboard', 'toaster'
    ]
    
    # Создаем SQL запрос для подсчета частоты каждого объекта
    select_columns = [f'SUM("{obj}") as `{obj}`' for obj in objects]
    query = f"""
    SELECT {','.join(select_columns)}
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    """
    
    params = [start_date, end_date]
    
    if themes != 'all' and themes:
        if isinstance(themes, str):
            themes = [themes]
        query += " AND video_theme IN ({})".format(','.join(['?']*len(themes)))
        params.extend(themes)
    
    # Получаем данные
    df = pd.read_sql(query, conn, params=params).transpose().reset_index()
    df.columns = ['object', 'count']
    
    # Сортируем и берем топ-10
    top_objects = df.sort_values('count', ascending=False).head(10)
    
    # Создаем график
    fig = px.bar(
        top_objects,
        x='object',
        y='count',
        title='Top 10 Most Frequent Objects in Videos',
        labels={'object': 'Object', 'count': 'Number of Videos'},
        color='count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Roboto'},
        yaxis_title="Number of Videos",
        coloraxis_colorbar=dict(title='Count')
    )
    
    return fig

# Callback для среднего количества просмотров по объектам
@app.callback(
    Output('views-by-objects', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('theme-filter', 'value')]
)
def update_views_by_objects(start_date, end_date, themes):
    objects = [
        'face', 'man', 'woman', 'kid', 'text_on_picutre', 'person', 'bottle', 'bowl', 'cup', 
        'toilet', 'toothbrush', 'kite', 'umbrella', 'car', 'cell phone', 'truck', 'handbag', 
        'hot dog', 'remote', 'tie', 'tv', 'cat', 'couch', 'dog', 'horse', 'refrigerator', 
        'banana', 'motorcycle', 'skateboard', 'bed', 'oven', 'dining table', 'teddy bear', 
        'clock', 'knife', 'pizza', 'airplane', 'laptop', 'sandwich', 'train', 'frisbee', 
        'sports ball', 'book', 'mouse', 'spoon', 'vase', 'bird', 'sink', 'traffic light', 
        'apple', 'scissors', 'bus', 'chair', 'bicycle', 'potted plant', 'cake', 'cow', 
        'surfboard', 'backpack', 'tennis racket', 'baseball bat', 'donut', 'boat', 
        'wine glass', 'broccoli', 'bench', 'suitcase', 'orange', 'keyboard', 'fire hydrant', 
        'sheep', 'fork', 'giraffe', 'carrot', 'microwave', 'zebra', 'elephant', 'bear', 
        'baseball glove', 'hair drier', 'parking meter', 'stop sign', 'skis', 'snowboard', 'toaster'
    ]
    
    # Создаем SQL запрос для подсчета средних просмотров по каждому объекту
    select_columns = []
    case_statements = []
    
    for obj in objects:
        case_statements.append(f"""
        CASE WHEN "{obj}" = 1 THEN playCount ELSE NULL END as `{obj}_views`,
        CASE WHEN "{obj}" = 1 THEN 1 ELSE NULL END as `{obj}_count`
        """)
    
    query = f"""
    SELECT {','.join(case_statements)}
    FROM videos_metadata_to_analyze
    WHERE createTime BETWEEN ? AND ?
    """
    
    params = [start_date, end_date]
    
    if themes != 'all' and themes:
        if isinstance(themes, str):
            themes = [themes]
        query += " AND video_theme IN ({})".format(','.join(['?']*len(themes)))
        params.extend(themes)
    
    # Получаем данные
    df = pd.read_sql(query, conn, params=params)
    
    # Вычисляем средние просмотры для каждого объекта
    object_stats = []
    for obj in objects:
        total_views = df[f"{obj}_views"].sum()
        count = df[f"{obj}_count"].sum()
        if count > 0:
            object_stats.append({
                'object': obj,
                'avg_views': total_views / count,
                'count': count
            })
    object_stats = pd.DataFrame(object_stats)
    object_stats = object_stats[object_stats['count'] >= 15]
    # Сортируем и берем топ-10 по средним просмотрам
    top_objects = object_stats.sort_values('avg_views', ascending=False).head(10)
    
    # Создаем график
    fig = px.bar(
        top_objects,
        x='object',
        y='avg_views',
        title='Top 10 Objects by Average Views',
        labels={'object': 'Object', 'avg_views': 'Average Views', 'count': 'Number of Videos'},
        hover_data=['count'],
        color='avg_views',
        color_continuous_scale='Plasma'
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Roboto'},
        yaxis_title="Average Views",
        coloraxis_colorbar=dict(title='Avg Views')
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)