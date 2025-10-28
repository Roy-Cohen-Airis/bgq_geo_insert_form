import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import html, Input, Output
from dash import dcc
from dash import dash_table
from dash.dash_table import DataTable
from pandas_gbq import read_gbq

import dash.exceptions as dash_exceptions
import random

import os
import re
import math
import ast
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import yt_dlp
import warnings
from openai import OpenAI
import base64
import io
import requests
import time
import dash_player
from dash import callback_context
from shapely.geometry import Point, Polygon
import datetime
from datetime import timedelta
from datetime import date

# Google API imports
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from oauth2client.client import GoogleCredentials
from google.oauth2 import service_account
from google.cloud import bigquery





warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

# --------------------------------------------------------------------------------------
# CONFIG / CREDS (load once)
# --------------------------------------------------------------------------------------
SERVICE_ACCOUNT_FILE = r"arabic-transcription-435113-c8120df00a35.json"
bq_service_account = r"C:\Users\roy\OneDrive\Desktop\ASR JSONS\Geo_Analysis\airis-rnd-machines-c514798d388f.json"


# Sheets & Drive: load once
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly',
          'https://www.googleapis.com/auth/drive']
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()
drive_service = build('drive', 'v3', credentials=creds)

# BigQuery: single cached client + constants
BQ_PROJECT = "airis-rnd-machines"
DATASET = "Sample_Data"
TABLE_GEO = f"{BQ_PROJECT}.{DATASET}.Geo_App_DB"
TABLE_CUTS = f"{BQ_PROJECT}.{DATASET}.Cuts_DB"

BQ_CREDS = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

def get_bq_client():
    return bigquery.Client()
 
BQ_CLIENT =get_bq_client()
# Cache schemas once
# ---- replace your TABLE_*_SCHEMA definitions with this ----

# Geo_App_DB schema (exact order doesn’t matter for insert_rows_json,
# but names MUST match and TIME/TIMESTAMP must be valid strings)
TABLE_GEO_SCHEMA = [
    "Index",
    "Cut_ID",
    "record_id",
    "Country",
    "City",
    "Links",
    "Title",
    "Coordinates",
    "Analyst",
    "Source",
    "Original_Duration",      # TIME
    "Start_Time",             # TIME
    "Finish_Time",            # TIME
    "Duration",               # TIME
    "Time_of_the_day",
    "Terrain",
    "Weather",
    "Video_quality",
    "Camera_tilt",
    "Distance_from_building",
    "Occluded",
    "Distortions",
    "Logos_and_text",
    "Comments",
    "TimeStamp",              # TIMESTAMP
    "Scene_Description",      # REPEATED STRING (list[str])
]

# Cuts_DB schema
TABLE_CUTS_SCHEMA = [
    "Index",
    "Cut_ID",
    "Country",
    "City",
    "Links",
    "Title",
    "Annotated_File_Name",
    "Cut_Start",          # TIME
    "Cut_Finish",         # TIME
    "Cut_Duration",       # TIME
    "Cut_Size",           # STRING
    "GCP_Bucket_URL",
    "Ignored",            # BOOLEAN
    "Validated_By",
    "Upload_Time",        # TIMESTAMP
    "Video_Size_OG",
    "Video Duration_OG",  
]

# --------------------------------------------------------------------------------------
# OPENAI (once)
# --------------------------------------------------------------------------------------
def load_api_key(filepath=r"gpt_key.txt"):
    with open(filepath, 'r') as file:
        return file.read().strip()

apikey = load_api_key()
clients = OpenAI(api_key=apikey)

# --------------------------------------------------------------------------------------
# UI Styles (unchanged)
# --------------------------------------------------------------------------------------
button_style1 = {
    "borderRadius": "24px",
    "width": "250px",
    "padding": "15px 25px",
    "position": "absolute",
    "bottom": "30px",
    "right": "30px",
    "background": "linear-gradient(to right, #4facfe, #00f2fe)",
    "border": "2px solid black",
    "fontWeight": "600",
    "fontSize": "40px",
    "font-weight": 'bold',
    "color": "white",
    "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)",
    "transition": "all 0.3s ease-in-out"
}

button_style10 = {
    "borderRadius": "24px",
    "width": "250px",
    "padding": "15px 25px",
    "position": "absolute",
    "bottom": "150px",
    "right": "30px",
    "background": "linear-gradient(to right, #006400, #98FB98)",
    "border": "2px solid black",
    "fontWeight": "600",
    "fontSize": "40px",
    "font-weight": 'bold',
    "color": "white",
    "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)",
    "transition": "all 0.3s ease-in-out"
}

img_disclamer_style = {
    'color':'red',
    'fontSize' : '24px',
    "font-weight": 'bold',
    "bottom": "150px",
    "right": "30px",
    "position": "absolute",
    'display':'none'    
    
    
}

button_style2 = {
    "borderRadius": "24px",
    "width": "250px",
    "padding": "15px 25px",
    "background": "linear-gradient(to right, #006400, #98FB98)",
    "border": "2px solid black",
    "fontWeight": "bold",
    "fontSize": "24px",
    "color": "white",
    "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)",
    "transition": "all 0.3s ease-in-out"
}
button_style3 = {
    "borderRadius": "24px",
    "width": "250px",
    "padding": "15px 25px",
    "background": "linear-gradient(to right, #B22222, #F08080)",
    "border": "2px solid black",
    "fontWeight": "bold",
    "fontSize": "24px",
    "color": "white",
    "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)",
    "transition": "all 0.3s ease-in-out",
}
button_style4 = {
    "padding": "5px 10px",
    "border": "2px solid black",
    "fontWeight": "bold",
    "fontSize": "24px",
    "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)",
    "position": "relative",
    "top": "350px"
}
rec_num = {"fontWeight": "bold", "fontSize": "18px", "textAlign": "center", "position": "relative", "top": "285px"}
save_link_btn = {
    "padding": "5px 10px", "border": "2px solid black", "fontWeight": "bold",
    "background-color": 'orange', "fontSize": "16px", "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)", "position": "relative", "top": "385px",
    "borderRadius": "50px",
}
link_check = {
    "padding": "5px 10px", "border": "2px solid black", "fontWeight": "bold",
    "background-color": 'purple', "fontSize": "16px", "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)", "position": "relative", "top": "400px",
    "borderRadius": "50px",
}
place_map_btn = {
    "padding": "5px 10px", "border": "2px solid black", "fontWeight": "bold",
    "background-color": 'blue', "fontSize": "16px", "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)", "position": "relative",
    "right": "100px", "top": "640px", "borderRadius": "50px",
}
check_location = {
    "padding": "5px 10px", "border": "2px solid white", "fontWeight": "bold",
    "background-color": 'black', "fontSize": "16px", "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)", "position": "relative",
    "top": "675px", "borderRadius": "50px",
}
check_btn_ed = {
    "padding": "5px 10px", "border": "2px solid black", "fontWeight": "bold",
    "background-color": 'blue', "fontSize": "16px", "textAlign": "center",
    "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.2)", "position": "relative",
    "top": "600px", "left": "-50px", "borderRadius": "50px",
}
check_div = {"top": "985px"}
background_style = {"background": "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)", "minHeight": "auto", "padding": "20px"}
container_style = {
    "backgroundColor": "white", "borderRadius": "30px", "padding": "30px",
    "boxShadow": "0 10px 40px rgba(0, 0, 0, 0.1)", "width": "100%", "maxWidth": "2400px",
    "margin": "auto", "position": "relative"
}
container_style_2 = container_style.copy()
heading_style = {
    "textAlign": "center", "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen",
    "color": "#333", "fontSize": "48px", "marginBottom": "30px"
}
heading_style2 = heading_style.copy(); heading_style2["fontSize"] = "28px"
tab_style = {"backgroundColor": "#E0FFFF", "borderRadius": "12px 12px 0 0", "padding": "10px 20px", "fontSize": "32px", "color": "#555", "border": "none", "fontWeight": "500"}
selected_tab_style = {"backgroundColor": "#00f2fe", "borderBottom": "2px solid #00f2fe", "fontWeight": "bold", "color": "white", "fontSize": "32px"}
modal_style = { "color": "#333", "fontSize": "20px", "padding": "10px", "fontWeight": "500" }
update_modal_style = {"textAlign": "center"}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Annotation Form"

# --------------------------------------------------------------------------------------
# HELPERS: Validation, parsing, small SQL checks, fast inserts
# --------------------------------------------------------------------------------------
# Precompiled regexes
_RE_NUMBERS = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')
_RE_YT = re.compile(
    r'^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)[\w-]{11}($|&)', re.IGNORECASE
)
_RE_IG = re.compile(r'^(https?://)?(www\.)?instagram\.com/(p|reel|tv)/[A-Za-z0-9_-]+/?$', re.IGNORECASE)
_RE_TME = re.compile(r'^(https?://)?(t\.me|telegram\.me)/[A-Za-z0-9_]{5,32}/?$', re.IGNORECASE)
_RE_FB = re.compile(r'^(https?://)?(www\.)?facebook\.com/(?:[^/?#&]+/)*[^/?#&]+/?$', re.IGNORECASE)
_RE_TK = re.compile(r'^(https?://)?(www\.)?tiktok\.com/@[A-Za-z0-9_.]+/video/\d+/?$', re.IGNORECASE)

def is_valid_coord(coord_str):
    numbers = _RE_NUMBERS.findall(str(coord_str))
    return len(numbers) >= 2

def clean_coordinate(coord_str: str) -> str:
    numbers = _RE_NUMBERS.findall(coord_str)
    if len(numbers) < 2:
        raise ValueError("Not enough coordinate numbers found in the input.")
    return f"{numbers[0]},{numbers[1]}"

def parse_input_time(time_str):
    """Accept HH:MM:SS or MM:SS; return timedelta."""
    parts = [p.strip() for p in str(time_str).split(':')]
    if len(parts) == 3:
        hh, mm, ss = map(int, parts)
    elif len(parts) == 2:
        hh = 0
        mm, ss = map(int, parts)
    else:
        raise ValueError("Expected format MM:SS or HH:MM:SS")
    return timedelta(hours=hh, minutes=mm, seconds=ss)

def parse_duration(duration_str: str) -> timedelta:
    parts = duration_str.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = map(int, parts)
    else:
        raise ValueError("Expected duration format MM:SS or HH:MM:SS")
    if not (0 <= minutes < 60 and 0 <= seconds < 60):
        raise ValueError("Minutes and seconds must be in the range 0-59")
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)




def is_valid_social_url(url: str) -> bool:
    import re
    patterns = {
        "youtube": re.compile(
            r'^(https?://)?(www\.)?'
            r'(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)'
            r'[\w-]{11}($|&)', re.IGNORECASE
        ),
        "instagram": re.compile(
            r'^(https?://)?(www\.)?instagram\.com/(p|reel|tv)/[A-Za-z0-9_-]+/?(\?.*)?$', re.IGNORECASE
        ),
        "telegram": re.compile(
            r'^(https?://)?(t\.me|telegram\.me)/[A-Za-z0-9_]{5,32}(/[\d]+)?/?$', re.IGNORECASE
        ),
        "facebook": re.compile(
            r'^(https?://)?(www\.)?facebook\.com/(?:[^/?#&]+/)*[^/?#&]+/?$', re.IGNORECASE
        ),
        "tiktok": re.compile(
            r'^(https?://)?(www\.)?tiktok\.com/@[A-Za-z0-9_.]+/video/\d+/?$', re.IGNORECASE
        ),
        "twitter": re.compile(
            r'^(https?://)?(www\.)?(x|twitter)\.com/[A-Za-z0-9_]{1,15}/status/\d+/?(\?.*)?$', re.IGNORECASE
        ),
    }

    return any(pattern.match(url) for pattern in patterns.values())

from google.cloud import storage
def upload_file_to_bucket(bucket_name, source_file_path, destination_blob_name):
    """
    Uploads a file to a GCP bucket.

    Args:
        bucket_name (str): Name of the target GCS bucket.
        source_file_path (str): Local path to the file to upload.
        destination_blob_name (str): Desired name/path for the file in the bucket.
                                    If None, uses the local filename.
    """
    # Initialize the GCS client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Default to using the local file name if no destination name is given
    if not destination_blob_name:
        destination_blob_name = os.path.basename(source_file_path)

    # Create a blob and upload the file
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
    
    print(f"File '{source_file_path}' uploaded to bucket '{bucket_name}' as '{destination_blob_name}'.")
    
    url = f"https://console.cloud.google.com/storage/browser/_details/{bucket_name}/{destination_blob_name};tab=live_object?inv=1&invt=Ab2zEA&project={bucket_name}"

    return url

def prefix_def (src):

    if src == 'Youtube':
        src_prefix = 'YT'
    elif src == 'Tiktok':
        src_prefix = 'TT'
    elif src == 'Telegram':
        src_prefix = 'TG'            
    elif src == 'facebook':
        src_prefix = 'FB'
    elif src == 'Instegram':
        src_prefix = 'IG' 
    elif src == 'other':
        src_prefix = 'NA'
    elif src == 'Twitter':
        src_prefix = 'TW'
    else:
        src_prefix = 'NONE' 
  
    return src_prefix    

def image_downloader (city_n,city_code,link,image_n):
    import os
    import requests
    from bs4 import BeautifulSoup
    
    city_code = str.lower(city_code)
    html = requests.get(link).text
    soup = BeautifulSoup(html, "html.parser")

    meta_data = {
        tag.get("property"): tag.get("content")
        for tag in soup.find_all("meta")
        if tag.get("property", "").startswith("og:")
    }

    image_url = meta_data.get("og:image")
    print("Image URL:", image_url)

    if image_url:
        response = requests.get(image_url)
        if response.status_code == 200:
            file_name = f"{image_n}.jpg"
            with open(file_name, "wb") as f:
                f.write(response.content)
            file_path = os.path.abspath(file_name)
            print("✅ Image downloaded successfully!")
            
            bucket_name = 'airis-rnd-benchmarks'
            blob_name = (f"geolocation/{city_code}/query-videos-annotated-cuts/{file_name}")            
            try:  
                upload_path =  upload_file_to_bucket(bucket_name,file_path,blob_name)           
                os.remove(file_path)
                return upload_path

            except Exception as e:
                os.remove(file_path)
                raise ValueError("Upload failed!")

        else:
            raise ValueError("Download failed") 
    else:
        raise ValueError("Download failed")
        
def general_validations(analyst,city_name,distancebuild,title,occlusion,terrain,
                        logo,distortions, tod,weather,vq,tilt,sources,scenes):
    if not analyst or analyst== "Select Analyst":
        raise ValueError("Please select an Analyst!")
    if not city_name:
        raise ValueError("Please select a City!")
    if not distancebuild or distancebuild=='Select a distance ':
        raise ValueError("Please Insert Distance from building!")
    if not title or title == 'Title Not Found! Please insert manually!':
        raise ValueError("Please Insert Title!")
    if not occlusion or occlusion == "Select an occlusion":
        raise ValueError("Please Insert Occlusion!")
    if not terrain or terrain == "Select a terrain":
        raise ValueError("Please Insert Terrain!")
    if not logo or logo == 'Select Logos & Text':
        raise ValueError("Please Insert Logos & Text!")
    if not distortions or distortions == 'Select a Distortion ':
        raise ValueError("Please Insert Distortions!")
    if not tod or tod == "Select time of day":
        raise ValueError("Please Insert Time of Day!")
    if not weather or weather == "Select weather ":
        raise ValueError("Please Insert Weather!")
    if not vq or vq== "Select video quality":
        raise ValueError("Please Insert Video Quality!")
    if not tilt or tilt == 'Select camera tilt ':
        raise ValueError("Please Insert Camera Tilt!")
    if not sources or sources == 'Select a source':
        raise ValueError("Please Insert Source!")
    if not scenes or scenes == 'Pick Scene':
        raise ValueError("Please Pick Scene/s!")



def is_valid_url (url):
    if not url:
        raise ValueError("Please insert a url!")
    else:
        valid_url = is_valid_social_url(url)
        if not valid_url:
            raise ValueError("Please insert a valid social url!")
        return url  

def is_valid_url_silent(url):
    if not url:
        return False
    valid_url = is_valid_social_url(url)
    return valid_url

def valid_coords(coords):
    if not coords:
        raise ValueError("Please insert coordinates!")
    if not is_valid_coord(coords):
        raise ValueError("Invalid Coordinates!")
    return clean_coordinate(coords)

def valid_dur(duration):
    if not duration:
        raise ValueError("Please insert duration!")
    if duration == "Invalid duration!":
        raise ValueError("Invalid Duration!")
    return duration

def parse_time_string(time_str):
    parts = list(map(int, time_str.strip().split(':')))
    if len(parts) == 3:
        hh, mm, ss = parts
    elif len(parts) == 2:
        hh = 0; mm, ss = parts
    elif len(parts) == 1:
        hh = mm = 0; ss = parts[0]
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    return hh, mm, ss

# Point-in-polygon
def is_inside_any(lat, lon, polygons):
    pt = Point(lat, lon)
    if polygons and Polygon(polygons).contains(pt):
        return True
    return False

from datetime import datetime, timezone

def to_time_str(hms: str) -> str:
    """Ensure HH:MM:SS format (TIME)."""
    parts = hms.split(":")
    if len(parts) == 2:
        return f"00:{int(parts[0]):02d}:{int(parts[1]):02d}"
    elif len(parts) == 3:
        hh, mm, ss = map(int, parts)
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    else:
        raise ValueError("TIME must be MM:SS or HH:MM:SS")

def now_timestamp_utc() -> str:
    """RFC3339 TIMESTAMP string BigQuery accepts."""
    return datetime.now(timezone.utc).isoformat()

# --------------------------------------------------------------------------------------
# BigQuery helpers (fast)
# --------------------------------------------------------------------------------------
def city_load_data(query: str) -> pd.DataFrame:
    try:
        query_job = BQ_CLIENT.query(query)
        df = query_job.result().to_dataframe(create_bqstorage_client=False)
    except Exception as e:
        print(f"❌ Error loading data from BigQuery: {e}")
        df = pd.DataFrame()
    return df

def get_city_bucket_code(city_name: str) -> str:
    query = """
        SELECT City_Bucket_CodeName
        FROM `airis-rnd-machines.Sample_Data.Cities DB`
        WHERE City_Name = @city_name
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("city_name", "STRING", city_name)
        ]
    )
    try:
        rows = BQ_CLIENT.query(query, job_config=job_config).result()
        for row in rows:
            return row.City_Bucket_CodeName
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def count_rows_by_city(city_name: str) -> int:
    """
    Count rows from Geo_App_DB where:
      - City = city_name
      - Annotated_File_Name contains 'img'
    """

    query = f"""
        SELECT COUNT(*) AS row_count
        FROM `{TABLE_CUTS}`
        WHERE City = @city_name
          AND Annotated_File_Name LIKE '%img%'
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("city_name", "STRING", city_name)
        ]
    )

    try:
        rows = BQ_CLIENT.query(query, job_config=job_config).result()
        for row in rows:
            return row.row_count
        return 0
    except Exception as e:
        raise ValueError(f"❌ Error counting rows for city '{city_name}': {e}")
        
        
    
def advanced_city_load(table: str, column: str, value) -> pd.DataFrame:
    try:
        query = f"SELECT * FROM `{table}` WHERE {column} = @value"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("value", "STRING", value)]
        )
        df = BQ_CLIENT.query(query, job_config=job_config).result().to_dataframe(create_bqstorage_client=False)
    except Exception as e:
        print(f"❌ Error loading data from BigQuery: {e}")
        df = pd.DataFrame()
    return df

def append_row_to_bq_cached(row_dict, table_id):
    """Fast streaming insert using cached client."""
    errors = BQ_CLIENT.insert_rows_json(table_id, [row_dict])
    print("✅ Row inserted into BigQuery.")
    if errors:
        raise RuntimeError(f"Insert error: {errors}")

def next_index(table_id: str) -> int:
    sql = f"SELECT IFNULL(MAX(Index), 0) + 1 AS idx FROM `{table_id}`"
    return list(BQ_CLIENT.query(sql))[0].idx

def link_coords_exists(link: str, coords: str) -> bool:
    sql = f"""
      SELECT COUNT(*) AS c
      FROM `{TABLE_GEO}`
      WHERE Links = @link AND Coordinates = @coords
    """
    job = BQ_CLIENT.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("link", "STRING", link),
            bigquery.ScalarQueryParameter("coords", "STRING", coords),
        ]
    ))
    return list(job)[0].c > 0

def link_timing_conflict(link: str, start_time_str: str, end_time_str: str) -> bool:
    sql = f"""
      WITH q AS (
        SELECT Start_Time, Finish_Time
        FROM `{TABLE_GEO}`
        WHERE Links = @link
      )
      SELECT COUNT(*) AS c
      FROM q
      WHERE @start BETWEEN TIME(Start_Time) AND TIME(Finish_Time)
         OR @end   BETWEEN TIME(Start_Time) AND TIME(Finish_Time)
         OR (TIME(Start_Time) BETWEEN @start AND @end)
         OR (TIME(Finish_Time) BETWEEN @start AND @end)
    """
    job = BQ_CLIENT.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("link", "STRING", link),
            bigquery.ScalarQueryParameter("start", "TIME", to_time_str(start_time_str)),
            bigquery.ScalarQueryParameter("end",   "TIME", to_time_str(end_time_str)),
        ]
    ))
    return list(job)[0].c > 0


def get_latest_geo_df():
    """Load the latest Geo_App_DB table from BigQuery."""
    query = f"SELECT City, Cut_ID, record_id, Links FROM `{TABLE_GEO}`"
    try:
        df = BQ_CLIENT.query(query).result().to_dataframe(create_bqstorage_client=False)
        return df
    except Exception as e:
        print(f"❌ Failed to load Geo_App_DB: {e}")
        return pd.DataFrame(columns=["City", "Cut_ID", "record_id", "Links"])


def generate_ids(city, df, url):
    """
    Generate (video_id, cut_id) exactly like the original generate_unique_random_id().
    city: str
    df: pd.DataFrame containing 'City', 'Cut_ID', 'record_id', and 'Links' columns
    url: str
    """
    df_filtered = df[df['City'] == city].copy()

    # Create a set of existing record_id prefixes for efficient lookup
    existing_prefixes = set(
        rec_id.split('_', 2)[0] + '_' + rec_id.split('_', 2)[1] + '_'
        for rec_id in df['Cut_ID'].values
    )
    
    
    def create_random_initials():
        country_name = cities[cities['City_Name'] == city]['Country'].values[0].lower()
        city_name = city.lower()
        if len(country_name) < 2 or len(city_name) < 3:
            raise ValueError("Country or city name too short for random sampling.")

        init_1 = ''.join(random.sample(country_name, 2))
        init_2 = ''.join(random.sample(city_name, 3))
        return f"{init_1}_{init_2}_"

    # Case 1: no records yet for this city
    if df_filtered.empty:
        country_name = cities[cities['City_Name'] == city]['Country'].values[0]
        init_1 = f"{country_name[:2].lower()}_"
        init_2 = f"{city[:3].lower()}_"
        initials = init_1 + init_2

        if initials in existing_prefixes:
            for _ in range(1000):
                initials = create_random_initials()
                if initials not in existing_prefixes:
                    break
            else:
                raise RuntimeError("Failed to find unique initials after many attempts.")

        xyz = f"{random.randint(9, 99)}{random.randint(9, 99)}{random.randint(99, 999)}"
        video_id = initials + xyz
        cut_id = f"{video_id}_cut1_v1"
        return video_id, cut_id

    # Case 2: this URL already has cuts in df → increment cut number
    df_filt2 = df[df['Links'] == url].copy()
    if not df_filt2.empty:
        df_filt2['base'] = df_filt2['Cut_ID'].str.extract(r'^(.*)_v\d+$')[0]
        df_filt2['version'] = df_filt2['Cut_ID'].str.extract(r'_v(\d+)$')[0].astype(int)

        latest_versions = (
            df_filt2.sort_values('version')
            .groupby('base', as_index=False)
            .last()
        )

        url_times = latest_versions['base'].nunique()
        extra_cut = url_times + 1
        new_cut_id = f"cut{extra_cut}"

        video_id = df_filt2['record_id'].values[0]
        cut_id = f"{video_id}_{new_cut_id}_v1"
        
    else:
    # Case 3: same city but new URL → reuse prefix
        rec_id = df_filtered['Cut_ID'].iloc[-1]

        rec_id_start = '_'.join(rec_id.split('_')[:2]) + '_'

        while True:
            xyz = f"{random.randint(9, 99)}{random.randint(9, 99)}{random.randint(99, 999)}"
            new_rec = rec_id_start + xyz
            if new_rec not in df_filtered['record_id'].values:
                video_id = new_rec
                cut_id = f"{video_id}_cut1_v1"
                return video_id, cut_id
    return video_id, cut_id
 
def download_upload_img(city_name,city_code,contents,file_n):
    if contents is None:
        raise ValueError("No file inserted")

    # Decode Base64 content
    header, encoded = contents.split(',')
    data = base64.b64decode(encoded)
    
    city_code = str.lower(city_code)
    # Optional: save file locally
    save_path = os.path.join('C:/Videos_Downloaded', file_n)
    with open(save_path, 'wb') as f:
        f.write(data)
        
    bucket_name = 'airis-rnd-benchmarks'
    blob_name = (f"geolocation/{city_code}/query-videos-annotated-cuts/{file_n}.jpg")
    try:  
        upload_path =  upload_file_to_bucket(bucket_name,save_path,blob_name)           
        os.remove(save_path)
        return upload_path

    except Exception as e:
        os.remove(save_path)
        raise ValueError("Download & Upload failed!")


def gcs_console_url_to_gs_url(url: str) -> str:
    # Match bucket and object path from the URL
    pattern = r"browser/_details/([^/]+)/(.+?)(?:;|$|\?)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid Google Cloud Storage console URL format")
    
    bucket, path = match.groups()
    return f"gs://{bucket}/{path}"
            
def display_media(url):
    # --- 1️⃣ Direct image files ---
    if re.search(r"\.(jpg|jpeg|png|gif|webp|svg)(\?|$)", url, re.IGNORECASE):
        return html.Img(
            src=url,
            style={
                "maxWidth": "600px",
                "border": "1px solid #ccc",
                "borderRadius": "8px"
            }
        )

    # --- 2️⃣ Instagram posts ---
    if "instagram.com/p/" in url or "instagram.com/reel/" in url:
        if 'utm_source=ig_embed' not in url:
            sep = '&' if '?' in url else '?'
            url = f"{url}{sep}utm_source=ig_embed"
        return html.Iframe(
            src=url,
            style={"width": "400px", "height": "480px", "border": "none"}
        )

    # --- 3️⃣ Twitter/X posts ---
    if "twitter.com" in url or "x.com" in url:
        return html.Iframe(
            src=url,
            style={"width": "550px", "height": "650px", "border": "none"}
        )

    # --- 4️⃣ Telegram posts ---
    # Works for links like: https://t.me/channelname/1234
    if "t.me/" in url:
        # Telegram provides embeddable widget for public channels
        embed_url = f"https://t.me/{'/'.join(url.split('/')[-2:])}?embed=1"
        return html.Iframe(
            src=embed_url,
            style={"width": "480px", "height": "600px", "border": "none"}
        )

    # --- 5️⃣ Facebook posts/photos/videos ---
    if "facebook.com" in url:
        # Convert post/video/photo URLs into embeddable iframe
        embed_url = f"https://www.facebook.com/plugins/post.php?href={url}"
        return html.Iframe(
            src=embed_url,
            style={"width": "500px", "height": "600px", "border": "none", "overflow": "hidden"}
        )

    # --- 6️⃣ Fallback ---
    return html.Div("Unsupported media source or invalid link.", style={"color": "red"})


def shorten_gcp_link(url: str):
    """
    Return a Dash html.A hyperlink where the text is the base filename
    (no extension), and the link points to the original GCP URL.
    """
    import urllib.parse, os
    # Strip query parameters and suffixes
    clean_url = url.split("?")[0].split(";")[0]

    # Extract filename
    filename = urllib.parse.unquote(clean_url.split("/")[-1])

    # Remove all extensions
    while True:
        base, ext = os.path.splitext(filename)
        if not ext:
            break
        filename = base

    # Return an html <a> element (Dash-friendly)
    return html.A(filename, href=url, target="_blank", style={"textDecoration": "underline", "color": "blue"})

    
    
# --------------------------------------------------------------------------------------
# INITIAL DATA (only once at startup)
# --------------------------------------------------------------------------------------
df_city_edit = city_load_data("SELECT * FROM `airis-rnd-machines.Sample_Data.Geo_App_DB`")
cities = city_load_data("SELECT * FROM `airis-rnd-machines.Sample_Data.Cities DB`")
cities_list = cities['City_Name'].unique()
country_list = cities['Country'].unique()

source_list = df_city_edit['Source'].unique()

time_list = df_city_edit['Time_of_the_day'].unique()
terrain_list = df_city_edit['Terrain'].unique()
weather_list = df_city_edit['Weather'].unique()
scene_description = ['Protests/Demonstrations','Crowd gathering in the street','March','Crime scene','Tourist tour','News interview','None']
video_vq = df_city_edit['Video_quality'].unique()
camera_tilt = df_city_edit['Camera_tilt'].unique()
distance = df_city_edit['Distance_from_building'].unique()
occlusion = df_city_edit['Occluded'].unique()
distortions = df_city_edit['Distortions'].unique()
logos = df_city_edit['Logos_and_text'].unique()
analysts = df_city_edit['Analyst'].unique()

cleaned_source = [v for v in source_list if v not in ("", None)]


cleaned_distance = [v for v in distance if v not in ("","Street level", None)]
cleaned_occlusion = [v for v in occlusion if v not in ("", None)]
cleaned_logos = [v for v in logos if v not in ("","slight ", "Prominent ", None)]
cleaned_distortions = [v for v in distortions if v not in ("","Motion DIstortions","No", None)]
cleaned_vq = [v for v in video_vq if v not in ("", None)]
cleaned_tilt = [v for v in camera_tilt if v not in ("", None)]
insert_mode = ['Manual','ChatGPT Suggestion']
file_type = ['Video','Image']

# --------------------------------------------------------------------------------------
# UI Layout (unchanged)
# --------------------------------------------------------------------------------------
def insert_tab_layout():
    return html.Div(
        style=background_style,
        children=[
            dbc.Container(
                style=container_style,
                children=[
                dcc.Store(id='default-values', data={
                    'link_url': "",
                    "coordinates_input": "",
                    'sources': "Select a source",
                    'file_type':'Video',
                    'input-hours': 0,
                    'input-minutes': 0,
                    'input-seconds': 0,
                    'input-hours_end': 0,
                    'input-minutes_end': 0,
                    'input-seconds_end': 0,
                    'tod': 'Select time of day',
                    'weather': "Select weather ",
                    'scene_desc': "Pick Scene",
                    'vq': "Select video quality",
                    'tilt': 'Select camera tilt ',
                    'distance': 'Select a distance ',
                    'occlusion_list': 'Select an occlusion',
                    'terrain': "Select a terrain",
                    'logos_list': 'Select Logos & Text',
                    'distortions_list': 'Select a Distortion ',
                    'analysts': 'Select Analyst',
                    'comments':""
                }),
                dcc.Store('links_table_store',data=None),
                dcc.Store('poly_store',data=None),
                dcc.Store(id="insert_guard", data={"pending": False, "cooldown_until": 0}),
                dcc.Store(id="insert_trigger", data=None),

                    html.H1("Geo Annotation Form", style=heading_style),
                    html.Hr(),
                    dbc.Row([
                        # First Column
                        dbc.Col([
                            html.H4("Link & Coordinates"),
                            dbc.Label("Choose a city:"),
                            dcc.Dropdown(
                                id='cities',
                                options=[{'label': k, 'value': k} for k in cities_list],
                                value="Rome",
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Country:"),
                            dcc.Dropdown(
                                id='country',
                                options=[{'label': k, 'value': k} for k in country_list],
                                value="Italy",
                                placeholder='country selection',
                                className="form-control",
                                disabled=True,
                            ),
                            html.Br(),
                            dbc.Label("Pick a source:"),
                            dcc.Dropdown(
                                id='sources',
                                options=[{'label': d, 'value': d} for d in cleaned_source],
                                value="",
                                placeholder = "Select a source",
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("File Type:"),
                            dcc.Dropdown(
                                id='file_type',
                                options=[{'label': d, 'value': d} for d in file_type],
                                value="Video",
                                placeholder = "Choose file type",
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Link:"),
                            dcc.Input(id='link_url', type='text', value="", className="form-control"),
                            html.Div(id="link_url_error", style={"color": "red"}),
                            html.Br(),
                            dbc.Label("Title:"),
                            dcc.Input(id='link_title', type='text',disabled=True, value="", className="form-control"),
                            html.Br(),
                            dbc.Label("Coordinates Insert Mode:"),
                            dcc.Dropdown(
                                id='insert_type',
                                options=[{'label': d, 'value': d} for d in insert_mode],
                                value='Manual',
                                placeholder = "Pick An Insert Mode",
                                className="form-control"
                            ),
                            dcc.Upload(
                                id='upload-image',
                                children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
                                style={
                                    'width': '100%', 'height': '60px','lineHeight': '60px',
                                    'borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px',
                                    'textAlign': 'center','margin': '10px','display': 'None'
                                },
                                multiple=False
                            ),
                            html.Div(id='output-image-upload'),
                            dbc.Label("Coordinates:"),
                            dcc.Input(id='coordinates_input', type='text', value="", className="form-control"),
                            html.Div(id="coords_error", style={"color": "red"}),
                            html.Br(),
                            dbc.Label("Auto-Generated Coordinates Location:"),
                            dcc.Input(id='gen_loc', type='text', disabled=True, className="form-control"),
                            html.Br(),
                            dbc.Label("Nearby Coordinates Radius (Meters):"),
                            dcc.Input(id='nearby', type='number', value=50 ,step=1, className="form-control"),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Div([
                            html.H4("Timing"),
                            dbc.Label("Start Time:"),
                            html.Br(),
                            html.Div([
                                html.Div([html.Label("Hours"), dcc.Input(id='input-hours', type='number', min=0, step=1, value=0, className="form-control")], style={'display': 'inline-block', 'margin-right': '10px'}),
                                html.Div([html.Label("Minutes"), dcc.Input(id='input-minutes', type='number', min=0, max=59, step=1, value=0, className="form-control")], style={'display': 'inline-block', 'margin-right': '10px'}),
                                html.Div([html.Label("Seconds"), dcc.Input(id='input-seconds', type='number', min=0, max=59, step=1, value=0, className="form-control")], style={'display': 'inline-block'}),
                            ]),
                            html.Br(),
                            dbc.Label("End Time:",style={'display':'block'}),
                            html.Br(),
                            html.Div([
                                html.Div([html.Label("Hours"), dcc.Input(id='input-hours_end', type='number', min=0, step=1, value=0, className="form-control")], style={'display': 'inline-block', 'margin-right': '10px'}),
                                html.Div([html.Label("Minutes"), dcc.Input(id='input-minutes_end', type='number', min=0, max=59, step=1, value=0, className="form-control")], style={'display': 'inline-block', 'margin-right': '10px'}),
                                html.Div([html.Label("Seconds"), dcc.Input(id='input-seconds_end', type='number', min=0, max=59, step=1, value=0, className="form-control")], style={'display': 'inline-block'}),
                            ]),
                            html.Br(),
                            html.Div([
                                html.Label("Cut Duration:", style={'fontWeight': 'bold', 'marginRight': '10px',}),
                                dcc.Input(id='output-duration', disabled=True, style={'width': '100px', 'fontWeight': 'bold'}),
                                dcc.Checklist(options=[{'label': '  Full', 'value': 'on'}], value=[], id='checkbox',
                                              style={'marginLeft': '20px', 'marginTop': '-10px'})
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                            html.Div([
                                html.Label("Full Duration:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                                dcc.Input(id='og_duration', disabled=False, style={'width': '100px', 'fontWeight': 'bold'}),
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                            html.Div(id="dur_error", style={"color": "red"})
                            ],id='timing_div',style={'display':'block'}),
                            html.Div([
                                html.Label("Full Size:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                                dcc.I
