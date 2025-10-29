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
                                dcc.Input(id='og_size', disabled=True, style={'width': '100px', 'fontWeight': 'bold'}),
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                        ], width=2),
                        dbc.Col([
                            dbc.Button("Save for later", id="save_later", color="success", n_clicks=0,style=save_link_btn),
                            dbc.Button("Link Check", id="link_check", color="success", n_clicks=0,style=link_check),
                            dbc.Button("Check", id="place_map", color="success", n_clicks=0,style=place_map_btn),
                            dbc.Button("Similar Locations?", id="f_loc", color="success", n_clicks=0,style=check_location)
                        ], width=1),
                        dbc.Col([
                            html.H4("Anchoring Features"),
                            dbc.Label("Distance from a building:"),
                            dcc.Dropdown(id='distance', options=[{'label': d, 'value': d} for d in cleaned_distance],
                                         value='', placeholder = "Select a distance", className="form-control"),
                            html.Br(),
                            dbc.Label("Occlusion:"),
                            dcc.Dropdown(id='occlusion_list', options=[{'label': d, 'value': d} for d in cleaned_occlusion],
                                         value='', placeholder = "Select an occlusion", className="form-control"),
                            html.Br(),
                            dbc.Label("Terrain type:"),
                            dcc.Dropdown(id='terrain', options=[{'label': d, 'value': d} for d in terrain_list],
                                         value="", placeholder = "Select a terrain", className="form-control"),
                            html.Br(),
                            dbc.Label("Logos and text:"),
                            dcc.Dropdown(id='logos_list', options=[{'label': d, 'value': d} for d in cleaned_logos],
                                         value='', placeholder = "Select Logos & Text", className="form-control"),
                            html.Br(),
                            dbc.Label("Distortions:"),
                            dcc.Dropdown(id='distortions_list', options=[{'label': d, 'value': d} for d in cleaned_distortions],
                                         value='', placeholder = "Select a Distortion", className="form-control"),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.H4(children=f"Map",style=heading_style2),
                            dl.Map(
                                id='map',
                                children=[dl.TileLayer(), dl.LayerGroup(id="map-layer", children=[]), dl.LayerGroup(id="polygon-layer", children=[])],
                                center=(41.9028, 12.4964),
                                zoom=6,
                                style={"width": "100%", "height": "400px", "margin": "6px","border": "2px solid black"}
                            ),
                        ], width=2),
                        dbc.Col(width=1),
                        dbc.Col([
                            html.H4("General Features"),
                            dbc.Label("Time of the day:"),
                            dcc.Dropdown(id='tod', options=[{'label': d, 'value': d} for d in time_list],
                                         value='', placeholder = "Select time of day", className="form-control"),
                            html.Br(),
                            dbc.Label("Weather:"),
                            dcc.Dropdown(id='weather', options=[{'label': d, 'value': d} for d in weather_list],
                                         value="", placeholder = "Select weather", className="form-control"),
                            html.Br(),
                            dbc.Label("Scene Description:"),
                            dcc.Dropdown(id='scene_desc', options=[{'label': d, 'value': d} for d in scene_description],
                                         value="", placeholder = "Pick Scene", multi=False, className="form-control"),
                            html.Br(),
                            dbc.Label("Quality:"),
                            dcc.Dropdown(id='vq', options=[{'label': d, 'value': d} for d in cleaned_vq],
                                         value="", placeholder = "Select video quality", className="form-control"),
                            html.Br(),
                            dbc.Label("Camera Tilt:"),
                            dcc.Dropdown(id='tilt', options=[{'label': d, 'value': d} for d in cleaned_tilt],
                                         value='', placeholder = "Select camera tilt", className="form-control"),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.H4("Analyst Data"),
                            dbc.Label("Anlyst:"),
                            dcc.Dropdown(id='analysts', options=[{'label': k, 'value': k} for k in analysts],
                                         placeholder="Select Analyst", className="form-control"),
                            html.Br(),
                            dbc.Label("Comments:"),
                            dcc.Input(id='comments', type='text', value="", className="form-control"),
                        ], width=2),
                        dbc.Col(width=1),
                        dbc.Col([
                            html.H4("Links Collection",style=heading_style2),
                            dash_table.DataTable(
                                id='links_table',
                                columns=[{'name': 'Links', 'id': 'links'}],
                                data=[],
                                row_selectable='single',
                                sort_action="native",
                                filter_action="native",
                                fixed_rows={'headers': True},
                                style_table={'maxHeight': '250px', 'overflowX': 'auto', 'overflowY': 'auto'},
                                style_cell={'textAlign': 'center', 'whiteSpace': 'normal', 'overflow': 'hidden',
                                            'textOverflow': 'clip', 'height': 'auto', 'width': '100px', 'maxWidth': '150px'},
                                style_header={'backgroundColor': 'rgb(30, 30, 30)','color': 'white','fontWeight': 'bold'},
                            ),
                            html.Br(), html.Br(), html.Br(),

                            html.Br(),
                            html.Div(
                                dash_player.DashPlayer(
                                    id='picked_video_insert',
                                    url="",
                                    controls=True,
                                    width="800px",
                                    height="400px",
                                    style={"border": "2px solid black"}
                                ),
                                style={"display": "flex","justifyContent": "center","marginBottom": "-50px"}
                            ),
                            html.Div(id='dynamic_image', style={"border": "2px solid black"}),
                            dcc.Upload(
                                id='upload_pic',
                                children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
                                style={
                                    'width': '100%', 'height': '60px','lineHeight': '60px',
                                    'borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px',
                                    'textAlign': 'center','margin': '10px','display': 'None'
                                },
                                multiple=False
                            ),
                            html.Div(
                                id='output_pic',
                                style={
                                    'display': 'none',
                                    'maxWidth': '400px',
                                    'maxHeight': '600px',
                                    'overflow': 'hidden'  # optional, in case the content exceeds limits
                                }
                            ),                    

                        ], width=3),
                        dbc.Col([
                            dbc.Button("Insert", id='insert', color='success', n_clicks=0, style=button_style1),
                            html.Div([
                                "Please keep in mind image inserting might take longer",
                                html.Br(),
                                "since it also downloads & uploads the data"
                            ], id='image_disclamer', style=img_disclamer_style),
                            dbc.Modal([dbc.ModalHeader("Annotation Details"), dbc.ModalBody(html.Div(id="confirmation-message", style=modal_style))],
                                      id="confirmation-modal", is_open=False),
                            dbc.Modal([dbc.ModalHeader("ChatGPT Results"), dbc.ModalBody(html.Div(id="gpt_res", style=modal_style))],
                                      id="gpt_modal", is_open=False),
                        ], width=2),
                    ])
                ]
            ),
        ]
    )

# --------------------------------------------------------------------------------------
# Clientside guard / cooldown (unchanged)
# --------------------------------------------------------------------------------------
app.clientside_callback(
    """
    function(n, guard) {
        if (!n) { return [window.dash_clientside.no_update, guard || {pending:false, cooldown_until:0}, false]; }
        const now = Date.now() / 1000;
        const g = guard || {pending:false, cooldown_until:0};
        if (g.pending || (g.cooldown_until && now < g.cooldown_until)) {
            return [window.dash_clientside.no_update, g, true];
        }
        return [now, {pending:true, cooldown_until:0}, true];
    }
    """,
    [Output("insert_trigger", "data"),
     Output("insert_guard", "data",allow_duplicate=True),
     Output("insert", "disabled",allow_duplicate=True)],
    [Input("insert", "n_clicks")],
    [State("insert_guard", "data")],
    prevent_initial_call='initial_duplicate'
)

# --------------------------------------------------------------------------------------
# Callbacks (unchanged except where noted)
# --------------------------------------------------------------------------------------
@app.callback(
    Output('upload-image', 'style'),
    Output('output-image-upload', 'style'),
    Output('output-image-upload', 'children'),
    Output('coordinates_input', 'value',allow_duplicate=True),
    Input("insert_type", "value"),
    State('upload-image', 'style'),
    State('output-image-upload', 'style'),
    prevent_initial_call='initial_duplicate'
)
def toggle_visibility(insert_input, upload_state, image_state):
    if upload_state is None: upload_state = {}
    if image_state is None: image_state = {}
    if insert_input:
        if insert_input == 'ChatGPT Suggestion':
            upload_state['display'] = 'block'
            image_state['display'] = 'block'
            return upload_state, image_state, html.Div(f""), ""
        else:
            upload_state['display'] = 'none'
            image_state['display'] = 'none'
            return upload_state, image_state, html.Div(f""), ""
    else:
        upload_state['display'] = 'none'
        return upload_state, dash.no_update, html.Div(f""), ""

@app.callback(
    Output('output-image-upload', 'children',allow_duplicate=True),
    Output('upload-image', 'contents', allow_duplicate=True),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    prevent_initial_call='initial_duplicate'
)
def update_output(content, filename):
    # 🧹 Clear upload immediately
    clear_contents = None  

    if content is None:
        return dash.no_update, clear_contents

    # Process normally
    preview = html.Div([
        html.H5(filename),
        html.Img(src=content, style={'width': '100%', 'height': 'auto'})
    ])

    # Return both the preview and the cleared upload state
    return preview, clear_contents



@app.callback(
    Output('output_pic', 'children',allow_duplicate=True),
    Output('upload_pic', 'contents', allow_duplicate=True),
    Input('upload_pic', 'contents'),
    State('upload_pic', 'filename'),
    prevent_initial_call='initial_duplicate'
)

def upload_special_sources(raw_data,name_file):
    clear_data = None  

    if raw_data is None:
        return dash.no_update, clear_data

    # Process normally
    preview_img = html.Div([
        html.H5(name_file),
        html.Img(src=raw_data, style={'width': '100%', 'height': 'auto'})
    ])

    # Return both the preview and the cleared upload state
    return preview_img, dash.no_update    





@app.callback(
    Output('coordinates_input', 'value',allow_duplicate=True),
    Output('gpt_modal','is_open'),
    Output('gpt_res','children'),
    Input('upload-image', 'contents'),
    State('cities', 'value'),
    State('upload-image', 'style'),
    prevent_initial_call='initial_duplicate'
)
def analyze_image(image_contents, city_name,upload_style):
    if upload_style.get('display') == 'block':
        if not image_contents or not city_name:
            return "Please provide both an image and a city name.", False, ""
        try:
            _, encoded = image_contents.split(",", 1)
        except Exception:
            return "Could not read image.", False, ""
        prompt = (
            f"Please analyze this image and provide the GPS coordinates "
            f"(latitude and longitude) of the exact location, assuming it is within the city of {city_name}. "
            f"Use high precision — **at least 6–7 decimal places**, like: 44.4493045, 26.086627. "
            f"If the precise location cannot be determined, estimate it based on visible landmarks, signs, or surroundings. "
            f"Return only the coordinates as two decimal values separated by a comma."
        )
        try:
            response = clients.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                    ]
                }],
                max_tokens=1000
            )
            content = response.choices[0].message.content
            match = re.search(r'(-?\d+\.\d+),\s*(-?\d+\.\d+)', content)
            if match:
                coords = f"{match.group(1)}, {match.group(2)}"
                gpt_results = html.Div(f"Raw model response: {content}")
                return coords, True, gpt_results
            else:
                gpt_results = html.Div(f"Raw model response: {content}")
                return "Could not extract coordinates.", True, gpt_results
        except Exception as e:
            return f"Error: {str(e)}", True, f"Failed process file!"
    else:
        return None, False, ""

@app.callback(
    Output('poly_store','data'),
    Output('polygon-layer', 'children'),
    Input("cities","value"),
)
def poly_extractor(city_val):
    if not city_val:
        return [],[]
    polygodid = cities[cities['City_Name'] == city_val]['PolygonID'].values[0]
    request = drive_service.files().get_media(fileId=polygodid)
    polygon_bytes = request.execute()
    try:
        if isinstance(polygon_bytes, bytes):
            polygon_data = json.loads(polygon_bytes.decode('utf-8'))
        else:
            polygon_data = json.loads(polygon_bytes)
    except Exception:
        polygon_data = []
    poly_coords = [tuple(coord) for coord in polygon_data]
    poly = Polygon(poly_coords)
    layer = dl.Polygon(positions=list(poly.exterior.coords), color="blue", fillColor="cyan", fillOpacity=0.6)
    polygon_layer = dl.LayerGroup(children=layer)
    return poly_coords, polygon_layer

@app.callback(
    Output('output-duration', 'value'),
    [
        Input('input-hours', 'value'),
        Input('input-minutes', 'value'),
        Input('input-seconds', 'value'),
        Input('input-hours_end', 'value'),
        Input('input-minutes_end', 'value'),
        Input('input-seconds_end', 'value'),
        Input('checkbox','value'),
        Input('og_duration', 'value'),
    ]
)
def calculate_duration(start_hours, start_minutes, start_seconds,
                       end_hours, end_minutes, end_seconds, checked, dur_og):
    if not checked:
        if None in (start_hours, start_minutes, start_seconds, end_hours, end_minutes, end_seconds):
            return "Invalid duration!"
        start_total = start_hours * 3600 + start_minutes * 60 + start_seconds
        end_total = end_hours * 3600 + end_minutes * 60 + end_seconds
        duration_diff = end_total - start_total
        if duration_diff <= 0:
            return "Invalid duration!"
        hours = duration_diff // 3600
        minutes = (duration_diff % 3600) // 60
        seconds = duration_diff % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        if not dur_og:
            return "Insert full video duration!"
        
        import datetime
        parsed_time = datetime.datetime.strptime(dur_og, "%H:%M:%S")
        return parsed_time.strftime("%H:%M:%S")

@app.callback(
    Output('input-hours', 'value',  allow_duplicate=True),
    Output('input-minutes', 'value',  allow_duplicate=True),
    Output('input-seconds', 'value',  allow_duplicate=True),
    Output('input-hours_end', 'value',  allow_duplicate=True),
    Output('input-minutes_end', 'value',  allow_duplicate=True),
    Output('input-seconds_end', 'value',  allow_duplicate=True),
    [Input('checkbox','value'), Input('og_duration', 'value')],
    prevent_initial_call='initial_duplicate'
)
def re_inputs(checked, full_dur):
    if checked:
        if full_dur:
            hours, minutes, seconds = map(int, full_dur.split(":"))
            return 0,0,0, hours, minutes, seconds
        else:
            return (dash.no_update,) * 6
    else:
        return 0,0,0,0,0,0

@app.callback(
    Output("confirmation-modal",   "is_open",  allow_duplicate=True),
    Output("confirmation-message",   "children",  allow_duplicate=True),
    Input("link_check", "n_clicks"),
    State("link_url", "value"),
    prevent_initial_call='initial_duplicate'
)
def check_link_exists(n_clicks, link):
    if not n_clicks or not link:
        return False, ""
    # small BQ exists check instead of pulling whole table
    sql = f"SELECT COUNT(*) AS c FROM `{TABLE_GEO}` WHERE Links=@link"
    cnt = list(BQ_CLIENT.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("link", "STRING", link)]
    )))[0].c
    if cnt and int(cnt) > 0:
        # fetch timings (short)
        sql2 = f"""
          SELECT Start_Time, Finish_Time
          FROM `{TABLE_GEO}`
          WHERE Links=@link
          ORDER BY Start_Time
        """
        rows = list(BQ_CLIENT.query(sql2, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("link", "STRING", link)]
        )))
        msg = html.Div([
            html.P("⚠️ Link already exists with the following timings:"),
            *[html.P(f"{i+1}. {r['Start_Time']} – {r['Finish_Time']}") for i, r in enumerate(rows)],
            html.P("Please choose a different timing.")
        ])
    else:
        msg = "✅ Link does not exist in the DB. Good to go!"
    return True, msg




@app.callback(
    Output("picked_video_insert", "style"),
    Output("timing_div", "style"),
    Output("dynamic_image", "style"),
    Output("upload_pic", "style"),
    Output("output_pic", "style"),
    Output("upload_pic", "contents", allow_duplicate=True),
    Output("upload_pic", "filename", allow_duplicate=True),
    Output("image_disclamer",'style'),
    
    Input("file_type", "value"),
    Input("sources", "value"),
    
    State("picked_video_insert", "style"),
    State("timing_div", "style"),
    State("dynamic_image", "style"),
    State("upload_pic", "style"),
    State("output_pic", "style"),
    State("image_disclamer",'style'),
    prevent_initial_call="initial_duplicate"
)
def toggle_video_or_image(file_type, src, player_state, timing_state, image_state, pic_upload, pic_output,disclamer_state):
    # Default style fallbacks
    if player_state is None: player_state = {}
    if timing_state is None: timing_state = {}
    if image_state is None: image_state = {}
    if pic_upload is None: pic_upload = {}
    if pic_output is None: pic_output = {}
    if disclamer_state is None: disclamer_state = {}

    # 🚀 Always reset upload contents + filename when callback fires
    clear_contents = ''
    clear_filename = ''

    # 🖼️ Image mode
    if file_type == 'Image':
        if 'Instegram' in src or 'other' in src:
            player_state['display'] = 'none'
            timing_state['display'] = 'none'
            image_state['display'] = 'none'
            pic_upload['display'] = 'block'
            pic_output['display'] = 'block'
            disclamer_state['display'] = 'block'
        else:
            player_state['display'] = 'none'
            timing_state['display'] = 'none'
            image_state['display'] = 'block'
            pic_upload['display'] = 'none'
            pic_output['display'] = 'none'
            disclamer_state['display'] = 'block'

    # 🎥 Video mode
    else:
        player_state['display'] = 'block'
        timing_state['display'] = 'block'
        image_state['display'] = 'none'
        pic_upload['display'] = 'none'
        pic_output['display'] = 'none'
        disclamer_state['display'] = 'none'

    return (
        player_state,
        timing_state,
        image_state,
        pic_upload,
        pic_output,
        clear_contents,
        clear_filename,
        disclamer_state
    )
    

 
    
    
@app.callback(
    Output("og_duration",            "value"),
    Output("dur_error",           "children"),
    Output("link_title",       "value"),
    Output("link_title",       "disabled"),
    Output("og_size",        "value"),

    
    Input('link_url','value'),
    Input('file_type','value'),

    prevent_initial_call=True,
)
def fetch_link_metadata(link,file_tp):
    import traceback
    import datetime

    """
    Populate original-duration, title, and size when the URL field changes.
    """
    
    
    # URL cleared  ──────────────────────────────────────────────────────────
    if not link:
        return "", "", "", True, ""

    # Fails basic URL validation  ──────────────────────────────────────────
    if not is_valid_url_silent(link):
        return "NA", "Invalid link", "", True, ""
    if file_tp == 'Video':
    # Try yt-dlp  ──────────────────────────────────────────────────────────
        try:
            with yt_dlp.YoutubeDL({}) as ydl:
                info = ydl.extract_info(link, download=False)

            secs        = info.get("duration", 0) or 0
            og_dur      = str(datetime.timedelta(seconds=secs))
            video_title = info.get("title", "No title found")

            size_bytes  = info.get("filesize") or info.get("filesize_approx")
            video_size  = f"{size_bytes / (1024 * 1024):.2f} MB" if size_bytes else ""

            return og_dur, "", video_title, True, video_size

        # Any failure in yt-dlp  ────────────────────────────────────────────────
        except Exception as e:
            print(f"❌ yt_dlp extraction failed for link: {link}")
            traceback.print_exc()  # prints the full stack trace
            return (
                "", f"Failed to extract video info: {e}",
                "Title Not Found! Please insert manually!", False, ""
            )
    else:
        import requests
        from bs4 import BeautifulSoup

        html = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        if "t.me/" in link:
            title_tag = soup.find("meta", property="og:description")
            raw_title = title_tag["content"] if title_tag else None
            post_title = raw_title

        elif "x.com/" in link or "twitter.com/" in link:
            # Extract Open Graph meta data (for title and image)
            og_tags = {tag.get("property"): tag.get("content") 
                    for tag in soup.find_all("meta", property=True)}

            post_title = og_tags.get("og:title")


            # Fallback: try old "description" meta tag if no og:title found
            if not post_title:
                title_tag = soup.find("meta", attrs={"name": "description"})
                raw_title = title_tag["content"] if title_tag else None
                if raw_title:
                    post_title = raw_title.split(" / X")[0].split(" on X")[0].strip()
                else:
                    post_title = f"Twitter post by {link.split('/')[3]}"
                    
        elif "facebook.com/" in link:

                title_tag = soup.find("meta", property="og:description")
                raw_title = title_tag["content"] if title_tag else None
                if raw_title:
                    post_title = raw_title.split(" / X")[0].split(" on X")[0].strip()
                else:
                    post_title = f"Facebook post by {link.split('/')[3]}"


        else:
            # Generic Open Graph handling (Instagram, etc.)
            title_tag = soup.find("meta", property="og:title")
            raw_title = title_tag["content"] if title_tag else None

            if raw_title:
                if ":" in raw_title:
                    post_title = raw_title.split(":", 1)[1].strip()
                else:
                    post_title = raw_title.strip()
                post_title = post_title.strip('"').strip()
            else:
                post_title = None

        # Thumbnail image
        img_tag = soup.find("meta", property="og:image")
        img_url = img_tag["content"] if img_tag else None


        # Get image size
        file_size = ""
        if img_url:
            head = requests.head(img_url, allow_redirects=True)
            size_bytes = int(head.headers.get("Content-Length", 0))
            file_size = round(size_bytes / (1024 * 1024), 2)
        img_size = f"{file_size} MB" if file_size else ""

        og_dur = "0:00:00"
        

        return og_dur,"",post_title,True,img_size  
    

         
@app.callback(
    [
        Output('confirmation-modal','is_open',allow_duplicate=True),
        Output('confirmation-message','children',allow_duplicate=True),
    ],
    [
        Input('coordinates_input','value'),
        Input('f_loc','n_clicks'),
        Input('nearby','value'),
        Input("cities","value")
    ],
    prevent_initial_call='initial_duplicate'
)
def check_famous_locations(f_coordinates, check_button, nearby, city_value):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    if triggered_id == 'f_loc.n_clicks':
        try:
            valid_f_coords = valid_coords(f_coordinates)
            lat, lon = map(float, valid_f_coords.split(","))
            if not nearby or nearby <= 0:
                raise ValueError("Invalid radius value, please make sure it's bigger than 0!")
            sql = """
                WITH loc AS (
                    SELECT Cut_ID, Coordinates
                    FROM `airis-rnd-machines.Sample_Data.Geo_App_DB`
                    WHERE City = @city
                ),
                cuts AS (
                    SELECT Cut_ID, GCP_Bucket_URL
                    FROM `airis-rnd-machines.Sample_Data.Cuts_DB`
                ),
                geo AS (
                    SELECT
                        l.Cut_ID,
                        c.GCP_Bucket_URL,
                        ST_GEOGFROMTEXT(
                            CONCAT('POINT(',
                                   TRIM(SPLIT(l.Coordinates, ',')[SAFE_OFFSET(1)]),' ',
                                   TRIM(SPLIT(l.Coordinates, ',')[SAFE_OFFSET(0)]),')')
                        ) AS geog
                    FROM loc l
                    LEFT JOIN cuts c USING (Cut_ID)
                ),
                distances AS (
                    SELECT
                        Cut_ID,
                        GCP_Bucket_URL,
                        ST_Y(geog) AS lat,
                        ST_X(geog) AS lon,
                        ST_Distance(geog, ST_GeogPoint(@lon, @lat)) AS distance_m
                    FROM geo
                    WHERE geog IS NOT NULL
                )
                SELECT Cut_ID, GCP_Bucket_URL, lat, lon, distance_m
                FROM distances
                WHERE distance_m <= @nearby
                ORDER BY distance_m ASC;
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("city", "STRING", city_value),
                    bigquery.ScalarQueryParameter("lat", "FLOAT64", lat),
                    bigquery.ScalarQueryParameter("lon", "FLOAT64", lon),
                    bigquery.ScalarQueryParameter("nearby", "FLOAT64", float(nearby)),
                ]
            )
            results = list(BQ_CLIENT.query(sql, job_config=job_config))
            coordinates_num = len(results)
            if coordinates_num == 0:
                found_message = f'No coordinates found within {nearby} meters from {valid_f_coords}!'
            else:
                found_message = html.Div([
                    html.P(f"✅ There are {coordinates_num} coordinate annotations within {nearby} meters around {valid_f_coords} in the following videos:"),
                    *[
                        html.P(children=[
                            f"{i+1}. ",
                            html.A(
                                href=row["GCP_Bucket_URL"],
                                children=os.path.basename(str(row["GCP_Bucket_URL"]).split("?")[0]),
                                target="_blank"
                            ),
                            f"  — {round(row['distance_m'], 1)} m"
                        ]) for i, row in enumerate(results)
                    ]
                ])
            return True, found_message
        except ValueError as e:
            error_message = html.Div([html.H5("⚠️ Validation Error", style={"color": "red"}), html.P(str(e), style={"color": "black"})])
            return True, error_message
        except Exception as e:
            error_message = html.Div([html.H5("🚨 Unexpected Error", style={"color": "red"}), html.P(str(e), style={"color": "black"})])
            return True, error_message
    else:
        return dash.no_update, dash.no_update

@app.callback(
    Output('scene_desc', 'multi'),
    Output('scene_desc', 'value',allow_duplicate=True),
    Input('scene_desc', 'value'),
    prevent_initial_call='initial_duplicate'
)
def toggle_multi(value):
    if "None" in value:
        return False, "None"
    else:
        return True, value

@app.callback(
    [
        Output('confirmation-modal','is_open'),
        Output('confirmation-message','children'),
        Output("country","value"),
        Output('link_url','value'),
        Output('coordinates_input','value'),
        Output('sources','value'),
        Output('file_type','value'),
        Output('input-hours','value'),
        Output('input-minutes','value'),
        Output('input-seconds','value'),
        Output('input-hours_end','value'),
        Output('input-minutes_end','value'),
        Output('input-seconds_end','value'),
        Output('tod','value'),
        Output('weather','value'),
        Output('vq','value'),
        Output('tilt','value'),
        Output('distance','value'),
        Output('occlusion_list','value'),
        Output('terrain','value'),
        Output('logos_list','value'),
        Output('distortions_list','value'),
        Output('analysts','value'),
        Output('comments','value'),
        Output('links_table','data'),
        Output('links_table_store','data'),
        Output('map', 'viewport'),
        Output('map-layer', 'children'),
        Output('checkbox','value'),
        Output('picked_video_insert','url'),
        Output('link_url_error', 'children'),
        Output('coords_error', 'children'),
        Output("insert", "disabled"),
        Output("insert_guard", "data"),
        Output('gen_loc','value'),
        Output('scene_desc','value'),
        Output('dynamic_image','children')
    ],
    [
        Input('insert_trigger','data'),
        Input("cities","value"),
        Input('cities', 'options'),
        Input("country","value"),
        Input('link_url','value'),
        Input('coordinates_input','value'),
        Input('sources','value'),
        Input('input-hours','value'),
        Input('input-minutes','value'),
        Input('input-seconds','value'),
        Input('input-hours_end','value'),
        Input('input-minutes_end','value'),
        Input('input-seconds_end','value'),
        Input('tod','value'),
        Input('weather','value'),
        Input('vq','value'),
        Input('tilt','value'),
        Input('distance','value'),
        Input('occlusion_list','value'),
        Input('terrain','value'),
        Input('logos_list','value'),
        Input('distortions_list','value'),
        Input('analysts','value'),
        Input('comments','value'),
        Input('save_later','n_clicks'),
        Input('links_table','selected_rows'),
        Input('place_map','n_clicks'),
        Input('scene_desc','value'),
        Input('map-layer', 'children'),
        Input('file_type','value'),
        Input('upload_pic', 'contents'),

    ],
    [
        State('default-values','data'),
        State('output-duration','value'),
        State('links_table_store','data'),
        State('checkbox','value'),
        State('og_duration', 'value'),
        State("link_title","value"),
        State("og_size","value"),
        State('poly_store','data'),
        State("insert_guard", "data"),
        State('upload_pic', 'filename'),
    ],
    prevent_initial_call=True,
)
def validations(insertbtn, city_name, city_options, country_name, linkurl, coords_input, sources,
                hourst, minst, secst, hourend, minend, secend, tod, weather, vq, tilt, distancebuild,
                occlusion, terrain, logos, distortions, analyst, comments, save_later_btn,
                selected_link, place_on_map, scene_desc, marker_map,file_tp,pic_contents,
                defaults, dur_input, links_table, checkbox, og_dur, title_vid, og_size_val, poly_tupple, insert_guard,pic_file_n):

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    


    # Normalize links_table
    if links_table is not None and (
        (isinstance(links_table, pd.DataFrame) and not links_table.empty)
        or (not isinstance(links_table, pd.DataFrame) and len(links_table))
    ):
        links_data = links_table
    else:
        links_data = []

    # --- quick reactions ---
    if triggered_id == 'link_url.value':
        if linkurl:
            try:
                #has to add ifs and 3 outputs at least
                valid_url_watch = is_valid_url(linkurl)
                if valid_url_watch:
                    if file_tp == 'Video':
                        picked_video = valid_url_watch
                        return (
                            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            dash.no_update,dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            dash.no_update, dash.no_update, dash.no_update,dash.no_update,picked_video,"",dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update   
                        )
                    else:
                        
                        img_output = display_media(valid_url_watch)
                        return (
                            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            dash.no_update,dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            dash.no_update, dash.no_update, dash.no_update,dash.no_update,"","",dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,img_output   
                        )                        
            except ValueError:
                error_input = html.Div(f"Incorrect Link Format", style={"color": "red"})
                return (
                dash.no_update, dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
                dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
                dash.no_update,dash.no_update,dash.no_update, dash.no_update,dash.no_update,dash.no_update,
                dash.no_update,dash.no_update, dash.no_update,dash.no_update,dash.no_update, dash.no_update , dash.no_update,    
                dash.no_update , dash.no_update,dash.no_update,dash.no_update,error_input,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update
                )
        else:
            return (
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update,dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update,dash.no_update,"","",dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,""
            )

    if triggered_id == 'coordinates_input.value':
        if coords_input:
            try:
                valid_coordintes_place = is_valid_coord(coords_input)
                if valid_coordintes_place or coords_input:
                    lat, lon = map(float, coords_input.split(","))
                    marker = dl.Marker(position=[lat, lon], children=[dl.Popup(coords_input)], id='city-mark')
                    viewport = {'center': [lat, lon], 'zoom': 14}
                    from geopy.geocoders import Nominatim
                    geolocator = Nominatim(user_agent="my_geo_app_roy")
                    location = geolocator.reverse((coords_input))
                    location_output = location.address if location else "Location Was Not Found!"
                    return (
                        dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update,dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update, viewport, [marker],dash.no_update,dash.no_update,dash.no_update,"",dash.no_update,dash.no_update,location_output,dash.no_update,dash.no_update
                    )
            except ValueError:
                error_input = html.Div(f"Incorrect Coordinates Format", style={"color": "red"})
                return (
                dash.no_update, dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
                dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
                dash.no_update,dash.no_update,dash.no_update, dash.no_update,dash.no_update,dash.no_update,
                dash.no_update,dash.no_update, dash.no_update,dash.no_update,dash.no_update, dash.no_update , dash.no_update,    
                dash.no_update , dash.no_update,dash.no_update,dash.no_update,dash.no_update,error_input,dash.no_update,dash.no_update,"",dash.no_update,dash.no_update
                )
        else:
            if city_name:
                center = cities[cities['City_Name'] == city_name]['CityCenter'].iloc[0]
                lat, lon = map(float, center.split(","))
                map_center_city = {'center': [lat, lon], 'zoom': 10}
            else:
                map_center_city = {'center': [41.9028, 12.4964], 'zoom': 10}
            return (
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update,dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, map_center_city,[],dash.no_update,dash.no_update,dash.no_update,"",dash.no_update,dash.no_update,"",dash.no_update,dash.no_update
            )

    elif triggered_id == 'insert_trigger.data':
        if file_tp == 'Video':
            try:
                from datetime import datetime
                import pytz

                if not insert_guard or not insert_guard.get("pending"):
                    raise PreventUpdate
                

                general_validation = general_validations(analyst, city_name, distancebuild, title_vid, occlusion, terrain,
                                                        logos, distortions, tod, weather, vq, tilt, sources, scene_desc)

                valid_url = is_valid_url(linkurl)
                valid_coordinates = valid_coords(coords_input)
                lat, lon = map(float, valid_coordinates.split(","))

                if not is_inside_any(lat, lon, poly_tupple):
                    raise ValueError("Make sure you're coordinates are in the polygon!")

                # compute times
                if checkbox is None or len(checkbox) == 0:
                    start_time = f"{hourst:02}:{minst:02}:{secst:02}"
                    end_time   = f"{hourend:02}:{minend:02}:{secend:02}"
                    h, m, s = map(int, dur_input.split(":"))
                    dur_time = f"{h:02}:{m:02}:{s:02}"
                else:
                    start_time = f"{hourst:02}:{minst:02}:{secst:02}"
                    end_time   = f"{hourend:02}:{minend:02}:{secend:02}"
                    dur_time   = dur_input

                # duration sanity
                time_obj = datetime.strptime(dur_time, "%H:%M:%S")
                dur_parse = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
                if dur_parse > 300:
                    raise ValueError("Pay Attention! Cut duration exceeding 5 minutes!")

                # end <= original duration
                og_dur_check = parse_duration(og_dur)
                end_time_check = parse_input_time(end_time)
                if end_time_check > og_dur_check:
                    raise ValueError("Annotation time exceeding original video duration!")

                # duplicate checks (fast SQL)
                if link_coords_exists(valid_url, valid_coordinates):
                    raise ValueError("Video link and Coordinates already exist!")
                if link_timing_conflict(valid_url, start_time, end_time):
                    raise ValueError("There's already this video with another crossing timing (or Full Video). Please select another duration!")

                # title (keep your logic)
                if 'youtube' in valid_url.lower() and 'Youtube' in str(sources):
                    video_name = title_vid
                elif 'tiktok' in valid_url.lower() and 'Tiktok' in str(sources):
                    video_name = title_vid
                elif 'facebook' in valid_url.lower() and 'facebook' in str(sources):
                    video_name = title_vid
                else:
                    video_name = title_vid  # fallback

                # timezone & timestamp
                local_tz = pytz.timezone("Asia/Jerusalem")
                formatted_datetime = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")

                # original duration normalized HH:MM:SS
                if og_dur:
                    parts = og_dur.split(":")
                    if len(parts) == 2:
                        formatted_dur = f"00:{parts[0]:0>2}:{parts[1]:0>2}"
                    elif len(parts) == 3:
                        formatted_dur = og_dur
                    else:
                        raise ValueError("Please insert original video duration in HH:MM:SS")
                else:
                    raise ValueError("Please insert original video duration")

                # fast next indices
                latest_index  = int(next_index(TABLE_GEO))
                latest_cut_ix = int(next_index(TABLE_CUTS))
                
                df_geo = get_latest_geo_df()

                # IDs without reading tables
                video_id, cut_id = generate_ids(city_name,df_geo,linkurl)

                if "None" in scene_desc:
                    scene_desc = [scene_desc]
                print(scene_desc)
                
            
                # Build row dicts with cached schema order
                # Geo row
                geo_values_by_name = {
                    "Index": latest_index,
                    "Cut_ID": cut_id,
                    "record_id": video_id,
                    "Country": country_name,
                    "City": city_name,
                    "Links": valid_url,
                    "Title": video_name,
                    "Coordinates": valid_coordinates,
                    "Analyst": analyst,
                    "Source": sources,
                    "Original_Duration": formatted_dur,
                    "Start_Time": start_time,
                    "Finish_Time": end_time,
                    "Duration": dur_time,
                    "Time_of_the_day": tod,
                    "Terrain": terrain,
                    "Weather": weather,
                    "Video_quality": vq,
                    "Camera_tilt": tilt,
                    "Distance_from_building": distancebuild,
                    "Occluded": occlusion,
                    "Distortions": distortions,
                    "Logos_and_text": logos,
                    "Comments": comments,
                    "TimeStamp": formatted_datetime,
                    "Scene_Description": scene_desc,
                }
                
                print(geo_values_by_name)
                geo_row = {k: geo_values_by_name.get(k) for k in TABLE_GEO_SCHEMA}

                # Cuts row
                cuts_values_by_name = {
                    "Index": latest_cut_ix,
                    "Cut_ID": cut_id,
                    "Country": country_name,
                    "City": city_name,
                    "Links": valid_url,
                    "Title": video_name,
                    "Annotated_File_Name": 'TBD',
                    "Cut_Start": start_time,
                    "Cut_Finish": end_time,
                    "Cut_Duration": dur_time,
                    "Cut_Size": 'TBD',
                    "GCP_Bucket_URL": 'TBD',
                    "Ignored": 'FALSE',
                    "Validated_By": 'og_user',
                    "Upload_Time": formatted_datetime,
                    "Video_Size_OG": og_size_val,
                    "Video Duration_OG": formatted_dur,
                }
                
                print(cuts_values_by_name)
                cuts_row = {k: cuts_values_by_name.get(k) for k in TABLE_CUTS_SCHEMA}

                # Insert both rows (two fast streaming calls)
                append_row_to_bq_cached(geo_row, TABLE_GEO)
                append_row_to_bq_cached(cuts_row, TABLE_CUTS)

                # UI updates
                marker = dl.Marker(position=[lat, lon], children=[dl.Popup(valid_coordinates)], id='city-mark')
                viewport = {'center': [lat, lon], 'zoom': 10}

                links_dframe = pd.DataFrame(links_table or [])
                if 'links' in links_dframe.columns:
                    links_dframe = links_dframe[links_dframe['links'] != valid_url]
                else:
                    links_dframe = pd.DataFrame(columns=['links'])
                links_data_clean = links_dframe.to_dict('records')

                now = time.time()
                guard_after_success = {"pending": False, "cooldown_until": now + 25}

                result_window = html.Div([
                    html.H1('Video Added Successfully!'),
                    html.Br(),
                    html.H3("Video Details: "),
                    html.Ul([
                        html.Li(f"City: {city_name}"),
                        html.Li(f"City: {country_name}"),
                        html.Li(f"Cut_id: {cut_id}"),
                        html.Li(f"Video Link: {valid_url}"),
                        html.Li(f"Video Name: {video_name}"),
                        html.Li(f"Video Source: {sources}"),
                        html.Li(f"Coordinates: {valid_coordinates}"),
                        html.Li(f"Start Time: {start_time}"),
                        html.Li(f"Finish Time: {end_time}"),
                        html.Li(f"Video Duration: {dur_time}"),
                        html.Li(f"Analyst: {analyst}"),
                        html.Li(f"Time of the day: {tod}"),
                        html.Li(f"Weather: {weather}"),
                        html.Li(f"Scene Description: {scene_desc}"),
                        html.Li(f"Video Quality: {vq}"),
                        html.Li(f"Camera Tilt: {tilt}"),
                        html.Li(f"Distance from a building: {distancebuild}"),
                        html.Li(f"Occlusion: {occlusion}"),
                        html.Li(f"Terrain: {terrain}"),
                        html.Li(f"Logos and Text: {logos}"),
                        html.Li(f"Distortions: {distortions}"),
                        html.Li(f"Comments: {comments}")
                    ])
                ])

                return (True, result_window,
                        dash.no_update,
                        defaults['link_url'],
                        defaults['coordinates_input'],
                        defaults['sources'],
                        defaults['file_type'],
                        defaults['input-hours'],
                        defaults['input-minutes'],
                        defaults['input-seconds'],
                        defaults['input-hours_end'],
                        defaults['input-minutes_end'],
                        defaults['input-seconds_end'],
                        defaults['tod'],
                        defaults['weather'],
                        defaults['vq'],
                        defaults['tilt'],
                        defaults['distance'],
                        defaults['occlusion_list'],
                        defaults['terrain'],
                        defaults['logos_list'],
                        defaults['distortions_list'],
                        defaults['analysts'],
                        defaults['comments'],
                        links_data_clean,
                        links_data_clean,
                        viewport,
                        [],
                        [],
                        "",  # picked_video url reset
                        dash.no_update,
                        dash.no_update,
                        False,
                        guard_after_success,
                        "",
                        defaults['scene_desc'],""
                )

            except ValueError as e:
                error_message = html.Div([html.H5("⚠️ Validation Error", style={"color": "red"}), html.P(str(e), style={"color": "black"})])
                guard_after_fail = {"pending": False, "cooldown_until": (insert_guard or {}).get("cooldown_until", 0)}
                return (True, error_message,dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update, dash.no_update , dash.no_update, dash.no_update ,
                dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,False,guard_after_fail,dash.no_update,dash.no_update,dash.no_update )
        else:
            try:
                from datetime import datetime
                import pytz

                if not insert_guard or not insert_guard.get("pending"):
                    raise PreventUpdate
                
                general_validation = general_validations(analyst, city_name, distancebuild, title_vid, occlusion, terrain,
                                                        logos, distortions, tod, weather, vq, tilt, sources, scene_desc)

                valid_url = is_valid_url(linkurl)
                valid_coordinates = valid_coords(coords_input)
                lat, lon = map(float, valid_coordinates.split(","))
                
                if "Instegram" in sources or 'other' in sources:
                    if not pic_contents:
                        raise ValueError("Please load a picture before inserting!")

                if not is_inside_any(lat, lon, poly_tupple):
                    raise ValueError("Make sure you're coordinates are in the polygon!")
                
                # title (keep your logic)
                if 'youtube' in valid_url.lower() and 'Youtube' in str(sources):
                    video_name = title_vid
                elif 'tiktok' in valid_url.lower() and 'Tiktok' in str(sources):
                    video_name = title_vid
                elif 'facebook' in valid_url.lower() and 'facebook' in str(sources):
                    video_name = title_vid
                elif 'instagram' in valid_url.lower() and 'instegram' in str(sources):
                    video_name = title_vid
                elif 'telegram' in valid_url.lower() and 'telegram' in str(sources):
                    video_name = title_vid
                else:
                    video_name = title_vid  # fallback

                # timezone & timestamp
                local_tz = pytz.timezone("Asia/Jerusalem")
                formatted_datetime = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S")

                # fast next indices
                latest_index  = int(next_index(TABLE_GEO))
                latest_cut_ix = int(next_index(TABLE_CUTS))
                
                df_geo = get_latest_geo_df()

                # IDs without reading tables
                video_id, cut_id = generate_ids(city_name,df_geo,linkurl)

                if "None" in scene_desc:
                    scene_desc = [scene_desc]
                print(scene_desc)

                # Build row dicts with cached schema order
                # Geo row
                geo_values_by_name = {
                    "Index": latest_index,
                    "Cut_ID": cut_id,
                    "record_id": video_id,
                    "Country": country_name,
                    "City": city_name,
                    "Links": valid_url,
                    "Title": video_name,
                    "Coordinates": valid_coordinates,
                    "Analyst": analyst,
                    "Source": sources,
                    "Original_Duration": None,
                    "Start_Time": None,
                    "Finish_Time": None,
                    "Duration": None,
                    "Time_of_the_day": tod,
                    "Terrain": terrain,
                    "Weather": weather,
                    "Video_quality": vq,
                    "Camera_tilt": tilt,
                    "Distance_from_building": distancebuild,
                    "Occluded": occlusion,
                    "Distortions": distortions,
                    "Logos_and_text": logos,
                    "Comments": comments,
                    "TimeStamp": formatted_datetime,
                    "Scene_Description": scene_desc,
                }
                
                print(geo_values_by_name)
                geo_row = {k: geo_values_by_name.get(k) for k in TABLE_GEO_SCHEMA}
                
                #downloading the image and uploading it to the right city bucket
                
                prefix = prefix_def(sources)
                city_code = get_city_bucket_code(city_name)
                next_img = (count_rows_by_city(city_name)) + 1

                image_name = f"{prefix}_img_{city_code}_v1_{next_img}_1"
                if 'Instegram' in sources or 'other' in sources:
                  bucket_url = download_upload_img(city_name,city_code,pic_contents,image_name)
                  UI_url= shorten_gcp_link(bucket_url)   
                  gs_path = gcs_console_url_to_gs_url(bucket_url) 
                else:
                   bucket_url = image_downloader(city_name,city_code,valid_url,image_name)
                   UI_url= shorten_gcp_link(bucket_url)                   
                   gs_path = gcs_console_url_to_gs_url(bucket_url)
                         
                # Cuts row
                cuts_values_by_name = {
                    "Index": latest_cut_ix,
                    "Cut_ID": cut_id,
                    "Country": country_name,
                    "City": city_name,
                    "Links": valid_url,
                    "Title": video_name,
                    "Annotated_File_Name": image_name,
                    "Cut_Start": None,
                    "Cut_Finish": None,
                    "Cut_Duration": None,
                    "Cut_Size": og_size_val,
                    "GCP_Bucket_URL": gs_path,
                    "Ignored": 'FALSE',
                    "Validated_By": 'og_user',
                    "Upload_Time": formatted_datetime,
                    "Video_Size_OG": og_size_val,
                    "Video Duration_OG": None,
                }

                print(cuts_values_by_name)
                cuts_row = {k: cuts_values_by_name.get(k) for k in TABLE_CUTS_SCHEMA}
                
                append_row_to_bq_cached(cuts_row, TABLE_CUTS)
                append_row_to_bq_cached(geo_row, TABLE_GEO)
                
                # UI updates
                marker = dl.Marker(
                    position=[lat, lon],
                    children=[dl.Popup(valid_coordinates)],
                    id='city-mark'
                )
                viewport = {'center': [lat, lon], 'zoom': 10}

                links_dframe = pd.DataFrame(links_table or [])
                if 'links' in links_dframe.columns:
                    links_dframe = links_dframe[links_dframe['links'] != valid_url]
                else:
                    links_dframe = pd.DataFrame(columns=['links'])
                links_data_clean = links_dframe.to_dict('records')

                now = time.time()
                guard_after_success = {"pending": False, "cooldown_until": now + 25}

                result_window = html.Div([
                    html.H1('Picture Added Successfully!'),
                    html.Br(),
                    html.H3("Picture Details: "),
                    html.Ul([
                        html.Li(f"City: {city_name}"),
                        html.Li(f"Country: {country_name}"),
                        html.Li(f"Cut_id: {cut_id}"),
                        html.Li(f"Picture Link: {valid_url}"),
                        html.Li(f"Picutre Name: {video_name}"),
                        html.Li(["GCP Link: ", UI_url]),
                        html.Li(f"Picture Source: {sources}"),
                        html.Li(f"Coordinates: {valid_coordinates}"),
                        html.Li(f"Analyst: {analyst}"),
                        html.Li(f"Time of the day: {tod}"),
                        html.Li(f"Weather: {weather}"),
                        html.Li(f"Scene Description: {scene_desc}"),
                        html.Li(f"Picutre Quality: {vq}"),
                        html.Li(f"Camera Tilt: {tilt}"),
                        html.Li(f"Distance from a building: {distancebuild}"),
                        html.Li(f"Occlusion: {occlusion}"),
                        html.Li(f"Terrain: {terrain}"),
                        html.Li(f"Logos and Text: {logos}"),
                        html.Li(f"Distortions: {distortions}"),
                        html.Li(f"Comments: {comments}")
                    ])
                ])

                return (
                    True,
                    result_window,
                    dash.no_update,
                    defaults['link_url'],
                    defaults['coordinates_input'],
                    defaults['sources'],
                    defaults['file_type'],
                    defaults['input-hours'],
                    defaults['input-minutes'],
                    defaults['input-seconds'],
                    defaults['input-hours_end'],
                    defaults['input-minutes_end'],
                    defaults['input-seconds_end'],
                    defaults['tod'],
                    defaults['weather'],
                    defaults['vq'],
                    defaults['tilt'],
                    defaults['distance'],
                    defaults['occlusion_list'],
                    defaults['terrain'],
                    defaults['logos_list'],
                    defaults['distortions_list'],
                    defaults['analysts'],
                    defaults['comments'],
                    links_data_clean,
                    links_data_clean,
                    viewport,
                    [],
                    [],
                    "",  # picked_video url reset
                    dash.no_update,
                    dash.no_update,
                    False,
                    guard_after_success,
                    "",
                    defaults['scene_desc'],
                    ""
                )

            except ValueError as e:
                error_message = html.Div([html.H5("⚠️ Validation Error", style={"color": "red"}), html.P(str(e), style={"color": "black"})])
                guard_after_fail = {"pending": False, "cooldown_until": (insert_guard or {}).get("cooldown_until", 0)}
                return (True, error_message,dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update, dash.no_update , dash.no_update, dash.no_update ,
                dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,False,guard_after_fail,dash.no_update,dash.no_update,dash.no_update )

            
    elif triggered_id == 'save_later.n_clicks':
        try:
            valid_url = is_valid_url(linkurl)
            if valid_url:
                if links_table is not None:
                    links_entry = next((entry for entry in links_table if entry.get('links') == valid_url), None)
                    if links_entry is None:
                        links_data.append({"links": f"{valid_url}"})
                    else:
                        links_data = links_table
                else:
                    links_data.append({"links": f"{valid_url}"})
            return (False,dash.no_update,dash.no_update,"",dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
            dash.no_update, dash.no_update,dash.no_update, dash.no_update,dash.no_update,dash.no_update,dash.no_update,
            dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
            dash.no_update, links_data ,links_data, dash.no_update , dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,"")
        except ValueError as e:
            error_message = html.Div([html.H5("⚠️ Validation Error", style={"color": "red"}), html.P(str(e), style={"color": "black"})])
            return (True, error_message,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
        dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
        dash.no_update,dash.no_update,dash.no_update, dash.no_update,dash.no_update,dash.no_update,
        dash.no_update,dash.no_update, dash.no_update,dash.no_update,dash.no_update, dash.no_update , dash.no_update,    
         dash.no_update , dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)

    elif triggered_id == 'links_table.selected_rows':
        row_idx = selected_link[0]
        links_df = pd.DataFrame(links_table)
        if row_idx < len(links_df):
            selected_url = links_df.iloc[row_idx][links_df.columns[0]]
            picked_url = selected_url
        return (False,dash.no_update,dash.no_update,selected_url,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
        dash.no_update, dash.no_update,dash.no_update, dash.no_update,dash.no_update,dash.no_update,dash.no_update,
        dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,
        dash.no_update, dash.no_update ,dash.no_update, dash.no_update , dash.no_update,dash.no_update,picked_url,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)

    elif triggered_id == 'place_map.n_clicks':
        try:
            validated_coor = valid_coords(coords_input)
            if not validated_coor or not city_name:
                raise ValueError('Please insert both coordinates & city name!')
            lat, lon = map(float, validated_coor.split(","))
            if is_inside_any(lat, lon, poly_tupple):
                validation_msg = html.Div([html.H5("✅ Success", style={"color": "green"}),
                                           html.P(f"{validated_coor} is in the Polygon", style={"color": "green", "font-weight": "bold"})])
            else:
                validation_msg = html.Div([html.H5("❌ Warning", style={"color": "red"}),
                                           html.P(f"{validated_coor} is out of the Polygon", style={"color": "red", "font-weight": "bold"})])
            marker = dl.Marker(position=[lat, lon], children=[dl.Popup(validated_coor)], id='city-mark')
            viewport = {'center': [lat, lon], 'zoom': 10}
            return (True, validation_msg, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update,dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,dash.no_update,
                viewport, [marker],dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update )
        except ValueError as e:
            error_message = html.Div([html.H5("⚠️ Validation Error", style={"color": "red"}), html.P(str(e), style={"color": "black"})])
            return (True, error_message, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update,dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)

    else:
        # city changed -> update country + map center
        if triggered_id == 'cities.value':
            if city_name:
                country_match = cities[cities['City_Name'] == city_name]['Country']
                country_val = country_match.iloc[0] if not country_match.empty else ''
                center = cities[cities['City_Name'] == city_name]['CityCenter'].iloc[0]
                lat, lon = map(float, center.split(","))
                map_center_city = {'center': [lat, lon], 'zoom': 10}
            else:
                country_val = ''
                map_center_city = {'center': [41.9028, 12.4964], 'zoom': 10}
            return (False,dash.no_update,country_val,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, map_center_city,
                [],dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)
        else:
            return (False, dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update ,
                    dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)

filters_list= ['City','Record ID','Analyst']        

# Find the first timestamp
df_city_edit['TimeStamp'] = pd.to_datetime(df_city_edit['TimeStamp'], errors='coerce')

# 2. Drop rows with bad timestamps if needed
df_city_edit = df_city_edit.dropna(subset=['TimeStamp'])

# 3. Get the earliest timestamp
first_timestamp = df_city_edit['TimeStamp'].min()


timeframes = ["", "Year", "1/2 Year", "3 Months", "Month", "Week", "Day", ""]

def edit_tab_layout():
     return html.Div(
        style=background_style,
        children=[
            dbc.Container(
                style=container_style_2,
                children=[
                dcc.Store(id='default-values_edit', data={
                    'link_url_edit': "",
                    "coordinates_input_edit": "",
                    'sources_edit': "",
                    'input-hours_edit': 0,
                    'input-minutes_edit': 0,
                    'input-seconds_edit': 0,
                    'input-hours_end_edit': 0,
                    'input-minutes_end_edit': 0,
                    'input-seconds_end_edit': 0,
                    'tod_edit': '',
                    'weather_edit': "",
                    'vq_edit': "",
                    'tilt_edit': '',
                    'distance_edit': '',
                    'occlusion_list_edit': '',
                    'terrain_edit': "",
                    'logos_list_edit': '',
                    'distortions_list_edit': '',
                    'analysts_edit': 'Select Analyst',
                    'comments_edit':""
                }),
                dcc.Store(id="stored-videoid", data=None),
                dcc.Store(id='latest_df',data=None),
                dcc.Store(id='latest_cut',data=None),

                    html.H1("Edit Mode", style=heading_style),
                    html.Hr(),

                    dbc.Row([
                        # First Column
                        dbc.Col([
                            html.H4("Pick a City & Video"),
                            dbc.Label("Choose a Filter:"),
                            dcc.Dropdown(
                                id='filters_list',
                                options=[{'label': k, 'value': k} for k in filters_list],
                                value="",
                                className="form-control",
                                placeholder = "Select a filter"
                            ),   
                            html.Br(),
                            dbc.Label("Filter Annotation Time by Last (All Time - Hour):"),
                            html.Div(
                                dcc.RangeSlider(
                                    id='timeframe_slider',
                                    min=0,
                                    max=len(timeframes) - 1,
                                    step=1,
                                    value=[0, len(timeframes) - 1],
                                    marks={i: label for i, label in enumerate(timeframes)},
                                    tooltip={"always_visible": True, "placement": "bottom"}
                                ),
                                style={"width": "550px", "margin-bottom": "20px"}  # Increase width as needed
                            ),
                            html.Br(),  
                            html.Br(),                       
                            dbc.Label("Choose a sub filter:"),
                            dcc.Dropdown(
                                id='cities_edit',
                                options=[{'label': k, 'value': k} for k in cities_list],
                                value="",
                                className="form-control"
                            ),
                            html.Br(),
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Pick a Cut ID:"),
                                    dcc.Dropdown(
                                        id='videoid_edit',
                                        options=[],
                                        value="",
                                        className="form-control",
                                        placeholder="Select a cut id"
                                    )
                                ], width=7),  # Half width
                                dbc.Col([
                                    dbc.Label("Pick a Version:"),
                                    dcc.Dropdown(
                                        id='cut_version',
                                        options=[],
                                        value="",
                                        className="form-control",
                                        placeholder="Select a version"
                                    )
                                ], width=5)  # Half width
                            ]),
                            html.Br()
                        ]),
                            dbc.Label("Pick a source:"),
                            dcc.Dropdown(
                                id='sources_edit',
                                options=[{'label': d, 'value': d} for d in source_list],
                                value="",
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Video Link:"),
                            dcc.Input(id='link_url_edit', type='text', value="", className="form-control"),
                            html.Div(id="link_error_edit", style={"color": "red"}),
                            html.Br(),
                            dbc.Label("Coordinates:"),
                            dcc.Input(id='coordinates_input_edit', type='text', value="", className="form-control"),
                            html.Div(id="coords_error_ed", style={"color": "red"}),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.H4("Timing"),
                            dbc.Label("Start Time:"),
                            html.Br(),
                            html.Div([
                                html.Div([
                                    html.Label("Hours"),
                                    dcc.Input(id='input-hours_edit', type='number', min=0, step=1, value=0, className="form-control"),
                                ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                html.Div([
                                    html.Label("Minutes"),
                                    dcc.Input(id='input-minutes_edit', type='number', min=0, max=59, step=1, value=0, className="form-control"),
                                ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                html.Div([
                                    html.Label("Seconds"),
                                    dcc.Input(id='input-seconds_edit', type='number', min=0, max=59, step=1, value=0, className="form-control"),
                                ], style={'display': 'inline-block'}),
                            ]),
                            html.Br(),
                            dbc.Label("End Time:"),
                            html.Br(),
                            html.Div([
                                html.Div([
                                    html.Label("Hours"),
                                    dcc.Input(id='input-hours_end_edit', type='number', min=0, step=1, value=0, className="form-control"),
                                ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                html.Div([
                                    html.Label("Minutes"),
                                    dcc.Input(id='input-minutes_end_edit', type='number', min=0, max=59, step=1, value=0, className="form-control"),
                                ], style={'display': 'inline-block', 'margin-right': '10px'}),
                                html.Div([
                                    html.Label("Seconds"),
                                    dcc.Input(id='input-seconds_end_edit', type='number', min=0, max=59, step=1, value=0, className="form-control"),
                                ], style={'display': 'inline-block'}),
                            ]),
                            html.Br(),
                            dbc.Label("Duration:  "),
                            dcc.Input(id='output-duration_edit', disabled=True, style={'margin-top': '30px', 'margin-left': '30px','font-weight': 'bold'}),
                            html.Div([
                                html.Label("Full Video Duration:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                                dcc.Input(id='og_dur_ed', disabled=False, style={'width': '100px', 'fontWeight': 'bold'}),
                                
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                            html.Div([
                                html.Label("Full Video Size:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                                dcc.Input(id='og_size_ed', disabled=True, style={'width': '100px', 'fontWeight': 'bold'}),
                                
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),                        

                        ], width=2),
                        dbc.Col([
                            html.Div([dbc.Label("Records Number:"),
                            html.Br(),
                            dcc.Input(id='rec_num', type='number', disabled=True, value=0, className="form-control")]
                            ,style=rec_num),
                            dbc.Button("↻", id='update_ids', color='success', n_clicks=0, style=button_style4),
                            dbc.Button("Check", id="place_map_ed", color="success", n_clicks=0,style=check_btn_ed),
                            dcc.Checklist(
                                    options=[{'label': '  Full', 'value': 'on'}],
                                    value=[], 
                                    id='checkbox_edit',
                                    style={'marginLeft': '-50px', 'marginTop': '965px'})                             
                            ],width=1),
                        dbc.Col([
                            html.H4("Anchoring Features"),
                            dbc.Label("Distance from a building:"),
                            dcc.Dropdown(
                                id='distance_edit',
                                options=[{'label': d, 'value': d} for d in distance],
                                value='',
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Occlusion:"),
                            dcc.Dropdown(
                                id='occlusion_list_edit',
                                options=[{'label': d, 'value': d} for d in occlusion],
                                value='',
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Terrain type:"),
                            dcc.Dropdown(
                                id='terrain_edit',
                                options=[{'label': d, 'value': d} for d in terrain_list],
                                value="",
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Logos and text:"),
                            dcc.Dropdown(
                                id='logos_list_edit',
                                options=[{'label': d, 'value': d} for d in logos],
                                value='',
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Distortions:"),
                            dcc.Dropdown(
                                id='distortions_list_edit',
                                options=[{'label': d, 'value': d} for d in distortions],
                                value='',
                                className="form-control"
                            ),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),      
                            html.H4("Map",id="map_title",style=heading_style2),
                            dl.Map(
                                id='map_edit',
                                children=[
                                    dl.TileLayer(),
                                    dl.LayerGroup(id="map-layer_ed", children=[]),
                                ],
                                center=(41.9028, 12.4964),  
                                zoom=10,
                                style={"width": "100%", "height": "400px", "margin": "6px","border": "2px solid black"}
                            ),
                        ], width=2),
                        dbc.Col(width=1),
                        dbc.Col([
                            html.H4("General Features"),
                            dbc.Label("Time of the day:"),
                            dcc.Dropdown(
                                id='tod_edit',
                                options=[{'label': d, 'value': d} for d in time_list],
                                value='',
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Weather:"),
                            dcc.Dropdown(
                                id='weather_edit',
                                options=[{'label': d, 'value': d} for d in weather_list],
                                value="",
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Video Quality:"),
                            dcc.Dropdown(
                                id='vq_edit',
                                options=[{'label': d, 'value': d} for d in video_vq],
                                value="",
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Camera Tilt:"),
                            dcc.Dropdown(
                                id='tilt_edit',
                                options=[{'label': d, 'value': d} for d in camera_tilt],
                                value='',
                                className="form-control"
                            ),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.H4("Analyst Data"),
                            dbc.Label("Anlyst:"),
                            dcc.Dropdown(
                                id='analysts_edit',
                                options=[{'label': k, 'value': k} for k in analysts],
                                placeholder="Select Analyst",
                                className="form-control"
                            ),
                            html.Br(),
                            dbc.Label("Comments:"),
                            dcc.Input(id='comments_edit', type='text', value="", className="form-control"),                           
                            ],width=2),
                        dbc.Col([html.H2("Watch It Here:"),
                    html.Br(),
                    html.Br(),    
                    html.Br(),                 
                html.Div(
                    dash_player.DashPlayer(
                        id='picked_video_edit',
                        url="",
                        controls=True,
                        width="800px",
                        height="400px",
                        style={"border": "2px solid black"}
                    ),
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "marginBottom": "-50px",
                    }
                ),
                
                html.Div([
                    html.Br(),
                    html.Div([
                        html.H2("To Full Dashboard", style={"textAlign": "center"}),  # added centered H2
                        html.A(
                            html.Img(
                                src="/assets/Full_Dashboard.png",
                                alt="To The Full Dashboard",  # added alt
                                style={
                                    'width': '500px',
                                    'border': '1px solid black'
                                }
                            ),
                            href='http://data-team-dashboard:8000/',
                            target='_blank'  # opens link in new tab
                        )
                    ])
                ],
                style={
                    "display": "flex",
                    "justifyContent": "right",
                    "gap": "70px",
                    "marginTop": "120px"
                })

  
                    ],width=4), 
                    html.Br(),
                    html.Div(
                        [
                            dbc.Button("Update", id='update', color='success', n_clicks=0, style=button_style2),
                            
                        ],
                        style={"display": "flex", "justifyContent": "right", "gap": "40px", "marginBottom": "30px"}
                    ),
                        # Third Column (Button + Modal)

                            dbc.Modal(
                                [
                                    dbc.ModalHeader("Edit Mode:"),
                                    dbc.ModalBody(
                                        html.Div(id="confirmation-message_edit", style=modal_style)
                                    ),
                                ],
                                id="confirmation-modal_edit",
                                is_open=False,
                            ),
                            dbc.Modal(
                                [
                                    dbc.ModalHeader("Confirmation"),
                                    dbc.ModalBody(
                                        html.Div("Are you sure you want to proceed?", style=modal_style)
                                    ),
                                    dbc.ModalFooter(
                                        dbc.ButtonGroup(
                                            [
                                                dbc.Button("Yes", id="confirm-yes", color="success", n_clicks=0),
                                                dbc.Button("No", id="confirm-no", color="danger", n_clicks=0),
                                            ],
                                            className="w-100",  # full width button group
                                        )
                                    ),
                                ],
                                id="confirmation-update",
                                is_open=False,
                            ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Removal Confirmation"),
                            dbc.ModalBody([
                                dbc.Label("Since it's a irreversable action, please insert the removal key: "),
                                dbc.Input(
                                    id="delete_password",
                                    type="password",
                                    placeholder="Enter your password...",
                                )
                        ]),
                            dbc.ModalFooter(
                                dbc.Button(
                                    "Delete",
                                    id="delete_btn",
                                    color="primary",
                                    className="ml-auto"
                                ),
                            ),
                        ],
                        id="delete-modal",
                        is_open=False,  # Initially closed
                    ),
                    ])
                ]
            ),
        ]
    )

def video_options_per_time (df,timing):
    today = pd.Timestamp.today().normalize()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
    time_delta = today - pd.Timedelta(days=timing)
    mask = df['TimeStamp'].between(time_delta, today)
    filtered_df = df.loc[mask].copy()
    filtered_df['base'] = filtered_df['Cut_iD'].str.extract(r'^(.*)_v\d+$')[0]
    voptions = filtered_df['base'].unique()
    
    return voptions
global time_constants
time_constants = ['Up to a week ago', 'Up to 2 weeks ago','Up to a month ago','Up to 3 months ago','Up to half a year ago', 'Up to a year ago', 'All Time'] 


@app.callback(
    Output('output-duration_edit', 'value'),
    [
        Input('input-hours_edit','value'),
        Input('input-minutes_edit','value'),
        Input('input-seconds_edit','value'),
        Input('input-hours_end_edit','value'),
        Input('input-minutes_end_edit','value'),
        Input('input-seconds_end_edit','value'),
        Input('checkbox_edit','value') 
    ]
)

def calculate_duration_edit(start_hours, start_minutes, start_seconds,
                       end_hours, end_minutes, end_seconds,checkbox_edit):
    # Calculate start and end times in total seconds
    if not checkbox_edit:
        # Ensure all start and end inputs are valid (not None)
        if start_hours is None or start_minutes is None or start_seconds is None:
            return "Invalid duration!"
        if end_hours is None or end_minutes is None or end_seconds is None:
            return "Invalid duration!"

        # Convert to total seconds
        start_total = start_hours * 3600 + start_minutes * 60 + start_seconds
        end_total = end_hours * 3600 + end_minutes * 60 + end_seconds

        duration_diff = end_total - start_total

        # Handle negative or zero duration
        if duration_diff <= 0:
            return "Invalid duration!"

        hours = duration_diff // 3600
        minutes = (duration_diff % 3600) // 60
        seconds = duration_diff % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    else:
        return "Full Video"


    
@app.callback(
    Output("cities_edit", "options"),
    Output("cities_edit", "value"),

    Output("latest_df",   "data"),
    Output("latest_cut",   "data"),

    Input("timeframe_slider", "value"),
    Input("filters_list", "value"),
    Input("update_ids",   "n_clicks"),
    
    prevent_initial_call='initial_duplicate'

)
def load_sub_filter(slider_val, selected_filter,update_ids):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    
    df_city_edit = city_load_data("SELECT * FROM `airis-rnd-machines.Sample_Data.Geo_App_DB`")
    # Assume slider_val is a list/tuple like [start_days_ago, end_days_ago]
    low_idx, high_idx = slider_val
    index_to_days = {
        7: 0,      # last hour? or 0 days? adjust as needed
        6: 1,      # last day
        5: 7,      # last week
        4: 30,     # last month
        3: 90,     # last 3 months
        2: 183,    # last 6 months
        1: 365,    # last year
        0: 1000    # "all time" or large number
    }

    now = pd.Timestamp.now(tz="Asia/Jerusalem")

    # map low_idx and high_idx separately
    start_cutoff = now - timedelta(days=index_to_days[high_idx])
    end_cutoff = now - timedelta(days=index_to_days[low_idx])

    df_city_edit['TimeStamp'] = pd.to_datetime(df_city_edit['TimeStamp'], utc=False, errors='coerce')        

    # Ensure timestamps are tz-aware
    if df_city_edit['TimeStamp'].dt.tz is None:
        df_city_edit['TimeStamp'] = df_city_edit['TimeStamp'].dt.tz_localize("Asia/Jerusalem")

    # Ensure start <= end
    if start_cutoff > end_cutoff:
        start_cutoff, end_cutoff = end_cutoff, start_cutoff

    # Filter using BETWEEN start and end
    filtered_df = df_city_edit[
        (df_city_edit['TimeStamp'] >= start_cutoff) &
        (df_city_edit['TimeStamp'] <= end_cutoff)
    ]

    # Return unique values based on selected filter
    if selected_filter == "City":
        opts = sorted(filtered_df["City"].dropna().unique())
    elif selected_filter == "Record ID":
        opts = sorted(filtered_df["record_id"].dropna().unique())
    elif selected_filter == "Analyst":
        opts = sorted(filtered_df["Analyst"].dropna().unique())
    else:
        opts = []
        
    if triggered_id == "update_ids.n_clicks":
        return (opts, "" , df_city_edit.to_dict('records'),city_load_data("SELECT * FROM `airis-rnd-machines.Sample_Data.Cuts_DB`").to_dict('records')) 
    else: 
        return (opts,dash.no_update, df_city_edit.to_dict('records'),city_load_data("SELECT * FROM `airis-rnd-machines.Sample_Data.Cuts_DB`").to_dict('records')) 
    

@app.callback(
    Output("videoid_edit","options"),
    Output("videoid_edit","value"),

    Output("rec_num","value"),

    Input("cities_edit",  "value"),
    State("latest_df",   "data"),

         # <-- NEW
    prevent_initial_call=True
)            

def loading_videoid_options(selected_input,latest_df):

    rec_num=""
    df = pd.DataFrame(latest_df)
    q2_df = df.copy()
    q2_df['base'] = q2_df['Cut_ID'].str.extract(r'^(.*)_v\d+$')[0]

    
    if selected_input in df_city_edit['City'].values:

        video_id_options =  q2_df[q2_df['City']==selected_input]['base'].unique()
        rec_num = q2_df[q2_df['City']==selected_input]['base'].shape[0]
    elif selected_input in q2_df['record_id'].values:
        video_id_options =  q2_df[q2_df['record_id']==selected_input]['base'].unique()
        rec_num = q2_df[q2_df['record_id']==selected_input]['base'].shape[0]
    elif selected_input in q2_df['Analyst'].values:
        video_id_options =  q2_df[q2_df['Analyst']==selected_input]['base'].unique() 
        rec_num = q2_df[q2_df['Analyst']==selected_input]['base'].shape[0] 
    else:
        video_id_options = []

    return (video_id_options,"Select a cut id",rec_num)

@app.callback(
    Output("cut_version","options"),
    Output("cut_version","value"),

    Input("videoid_edit","value"),

    State("latest_df",   "data"),
    State("cities_edit",  "value"),
    
)


def load_versions(cut,df,selected_vid):
    if selected_vid:
        v_df=pd.DataFrame(df)
        if cut:
            cuts_df = v_df[v_df['Cut_ID'].str.contains(cut)].copy()
            cuts_df['version'] = cuts_df['Cut_ID'].str.extract(r'_(v\d+)$')[0]
            version_options = sorted(cuts_df['version'].unique(), key=lambda x: int(x[1:]))  # remove 'v' and sort by int

            ver_val = ""
        else:
            version_options =[]
            ver_val = ""

        return version_options,ver_val
    return [],""

def video_info_extraction(link):
            ydl_opts = {}
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info_dict = ydl.extract_info(link, download=False)
                    video_size = info_dict.get('filesize') or info_dict.get('filesize_approx')
                    og_dur = info_dict.get('duration', 0)
                    if video_size  and og_dur:
                        vid_size = f"{video_size / (1024 * 1024):.2f} MB"
                        video_og_dur = str(timedelta(seconds=og_dur))
                        return vid_size,video_og_dur
                    else:
                        vid_size = 0
                        video_og_dur = 0
                    return vid_size,video_og_dur
                except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError, Exception) as e:
                    return "", ""
                    
@app.callback(
    Output('og_size_ed','value'),
    Output('og_dur_ed','value'),
    Input('link_url_edit','value'),
    State("videoid_edit","value"),
    State("cut_version","value"),
    State('latest_df','data'),
    State("latest_cut",   "data")
)



def cal_size (link,cut,ver,latest_df,latest_cut):
    from datetime import timedelta

    df = pd.DataFrame(latest_df)
    cuts = pd.DataFrame(latest_cut)


    if link:
        cut_vers = f"{cut}_{ver}"
        match_link = df[df['Cut_ID']==cut_vers]['Links'].values[0]
        if not match_link:
            vid_size,video_og_dur = video_info_extraction(match_link)
            return vid_size,video_og_dur
        else:
            match_dur = cuts[cuts['Cut_ID']==cut_vers]['Video Duration_OG'].values[0]
            match_size = cuts[cuts['Cut_ID']==cut_vers]['Video_Size_OG'].values[0]
            if match_dur and not match_size:
                vid_size,video_og_dur = video_info_extraction(match_link)
                if vid_size:
                    return vid_size,match_dur
                else:
                    return None, match_dur  
            elif not match_dur and match_size:
                vid_size,video_og_dur = video_info_extraction(match_link)
                if video_og_dur:
                    return match_size,video_og_dur
                else:
                    return match_size,None
            elif not match_dur and not match_size:
                vid_size,video_og_dur = video_info_extraction(match_link)
                if vid_size and video_og_dur :
                    return vid_size,video_og_dur
                else:
                    return None, None
            else:
                return match_size,match_dur
    else:
        return "", ""
    
@app.callback ([
        Output('confirmation-modal_edit','is_open'),
        Output('confirmation-message_edit','children'),
        Output("stored-videoid", "data"), 
        Output('sources_edit','value'),    
        Output('link_url_edit','value'),
        Output('coordinates_input_edit','value'),
        Output('input-hours_edit','value'),
        Output('input-minutes_edit','value'),
        Output('input-seconds_edit','value'),
        Output('input-hours_end_edit','value'),
        Output('input-minutes_end_edit','value'),
        Output('input-seconds_end_edit','value'), 
        Output('tod_edit','value'),
        Output('weather_edit','value'),       
        Output('vq_edit','value'),    
        Output('tilt_edit','value'),     
        Output('distance_edit','value'),     
        Output('occlusion_list_edit','value'), 
        Output('terrain_edit','value'),   
        Output('logos_list_edit','value'),
        Output('distortions_list_edit','value'),
        Output('analysts_edit','value'),
        Output('comments_edit','value'), 
        Output('picked_video_edit','url'),
        Output("confirmation-update", "is_open"),
        Output('delete-modal',"is_open"),
        Output('checkbox_edit','value'),
        Output('link_error_edit', 'children'),
        Output('coords_error_ed', 'children'),
        Output('map_edit', 'viewport'),
        Output('map-layer_ed', 'children'),
        Output("cities_edit",  "value" ,allow_duplicate=True),
        Output("latest_df",   "data",allow_duplicate=True),
        Output("latest_cut",   "data",allow_duplicate=True)
        
        
   ],
    [ 
    Input("cities_edit",  "value"),
    Input("videoid_edit","value"),
    Input("cut_version","value"),
    Input('sources_edit','value'),    
    Input('link_url_edit','value'),
    Input('coordinates_input_edit','value'),
    Input('tod_edit','value'),
    Input('weather_edit','value'),       
    Input('vq_edit','value'),    
    Input('tilt_edit','value'),     
    Input('distance_edit','value'),     
    Input('occlusion_list_edit','value'), 
    Input('terrain_edit','value'),   
    Input('logos_list_edit','value'),
    Input('distortions_list_edit','value'),
    Input('analysts_edit','value'),
    Input('comments_edit','value'),
    Input('update','n_clicks'),
    Input("confirm-yes", "n_clicks"),
    Input("confirm-no", "n_clicks"),
    Input('place_map_ed','n_clicks'),
    

           
    ],
[    
 State('output-duration_edit','value'), 
 State('default-values_edit','data'),
 State("stored-videoid", "data"),
 State('latest_df','data'),
 State("confirmation-update", "is_open"),
 State('delete_password','value'),
 State('checkbox_edit','value'),
State('input-hours_edit','value'),
State('input-minutes_edit','value'),
State('input-seconds_edit','value'),
State('input-hours_end_edit','value'),
State('input-minutes_end_edit','value'),
State('input-seconds_end_edit','value'), 
State("og_dur_ed","value"),
State('og_size_ed','value'),
State("latest_cut",   "data"),
State("cut_version","options"),  
],
 prevent_initial_call='initial_duplicate'
               
)
def edit_mode(city_name_edit, video_cut, video_version, sourceedit, linkedit, coord_edit, tod_edit, weather_edit, vq_edit, tilt_edit, distance_edit,
              occlusion_edit, terrain_edit, logos_edit, distortions_edit, analyst_edit, comments_edit, update,confirm_yes,confirm_no,
             check_btn,duration_edit, defaults_edit,stored_videoid,latest_df,update_confirmation,delete_password,checkbox_ed,
              hours_st_edit, minute_st_edit, sec_st_edit,hours_end_edit, min_end_edit, sec_end_edit,og_dur_val,sized,latest_cut,ver_ops):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    df_city_edit = pd.DataFrame(latest_df)
    cut_db_ed = pd.DataFrame(latest_cut)
    # Define fallback/default output with the right length (25 in your case)

    if video_version:
        video_ver = f"{video_cut}_{video_version}"
        city_val = df_city_edit[df_city_edit['Cut_ID'] == video_ver]['City'].values[0]   
    else:
        video_ver = ""
        city_val = ""
    print(video_ver)

    
    if not video_version :
        # Reset all fields to defaults
        return (
            False, dash.no_update, dash.no_update,"","","",0,0,0,0,0,0,"","","","","","","","","","","","",False,False,dash.no_update,dash.no_update,dash.no_update
        ,dash.no_update,[],dash.no_update,dash.no_update,dash.no_update)


                        
    elif triggered_id == 'link_url_edit.value':
        if linkedit:
            try: 
                valid_url_watch = is_valid_url(linkedit)
                if valid_url_watch:
                    picked_video = valid_url_watch
                    return (False, dash.no_update,stored_videoid) + (dash.no_update,) * 20 + (picked_video,False,False,dash.no_update,"",
                    dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)
            except ValueError as e:
                # If any validation fails, catch and show the error message
                error_input = html.Div(f"Incorrect Link Format", style={"color": "red"})   
                return (False, dash.no_update,stored_videoid) + (dash.no_update,) * 20 + (dash.no_update,False,False,
                dash.no_update,error_input,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)
        else:
            return (False, dash.no_update,stored_videoid) + (dash.no_update,) * 20 + ("",False,False,dash.no_update,"",dash.no_update,dash.no_update,
            dash.no_update,dash.no_update,dash.no_update,dash.no_update)
    elif triggered_id == 'coordinates_input_edit.value':
        if coord_edit:
            try: 
                lat_str, lon_str = coord_edit.split(",")
                # 2) … and 2) both must convert to float
                lat, lon = float(lat_str), float(lon_str)
                marker = dl.Marker(position=[lat, lon],
                                children=[dl.Popup(coord_edit)],
                                id='city-mark')
                viewport = {'center': [lat, lon], 'zoom': 14}
                return (False, dash.no_update,stored_videoid) + (dash.no_update,) * 20 + (dash.no_update,False,False,
                dash.no_update,dash.no_update,"",viewport,[marker],dash.no_update,dash.no_update,dash.no_update)
            except ValueError as e:
                # If any validation fails, catch and show the error message
                error_input_cor = html.Div(f"Incorrect Coordinates Format", style={"color": "red"})
                return (False, dash.no_update,stored_videoid) + (dash.no_update,) * 20 + (dash.no_update,False,False,
                dash.no_update,dash.no_update,error_input_cor,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)
        else:
            center= cities[cities['City_Name'] == city_name_edit]['CityCenter'].iloc[0]
            lat, lon =map(float, center.split(",")) 
            map_center_city = {'center': [lat, lon], 'zoom': 10}      
            return (False, dash.no_update,stored_videoid) + (dash.no_update,) * 20 + (dash.no_update,False,False,dash.no_update,dash.no_update
            ,"",map_center_city,[],dash.no_update,dash.no_update,dash.no_update)


    elif triggered_id == 'place_map_ed.n_clicks':
        try:
            lat_str, lon_str = coord_edit.split(",")
            # 2) … and 2) both must convert to float
            lat, lon = float(lat_str), float(lon_str)
            if not coord_edit or not city_name_edit:
                raise ValueError('Please insert both coordinates & city name!')
            
            polygodid = cities[cities['City_Name'] == city_name_edit]['PolygonID'].values[0]
            request = drive_service.files().get_media(fileId=polygodid)
            polygon_bytes = request.execute()

            try:
                if isinstance(polygon_bytes, bytes):
                    polygon_data = json.loads(polygon_bytes.decode('utf-8'))
                else:
                    polygon_data = json.loads(polygon_bytes)
            except Exception:
                polygon_data = []
            poly_coords = [tuple(coord) for coord in polygon_data]
   

            if is_inside_any(lat, lon, poly_coords):
                validation_msg = html.Div(
                    [
                        html.H5("✅ Success", style={"color": "green"}),
                        html.P(f"{coord_edit} is in the Polygon", style={"color": "green", "font-weight": "bold"})
                    ]
                )
            else:
                validation_msg = html.Div(
                    [
                        html.H5("❌ Warning", style={"color": "red"}),
                        html.P(f"{coord_edit} is out of the Polygon", style={"color": "red", "font-weight": "bold"})
                    ]
                )

            marker = dl.Marker(
                position=[lat, lon],
                children=[dl.Popup(coord_edit)],
                id='city-mark'
            )

            viewport = {'center': [lat, lon], 'zoom': 14}
            return (True, validation_msg,stored_videoid) + (dash.no_update,) * 20 + (dash.no_update,False,False,dash.no_update,
            dash.no_update,"",viewport,[marker],dash.no_update,dash.no_update,dash.no_update)


        except ValueError as e:
            error_message = html.Div(
                [
                    html.H5("⚠️ Validation Error", style={"color": "red"}),
                    html.P(str(e), style={"color": "black"})
                ]
            )
            return (True, validation_msg,stored_videoid) + (dash.no_update,) * 20 + (dash.no_update,False,False,dash.no_update,dash.no_update,
            "",viewport,[marker],dash.no_update,dash.no_update,dash.no_update)

    
    elif  triggered_id == 'cut_version.value':
        row = df_city_edit[df_city_edit['Cut_ID'] == video_ver]
        if not row.empty:
            time_st = row['Start_Time'].values[0]
            hh_st, mm_st, ss_st = parse_time_string(time_st)

            time_end = row['Finish_Time'].values[0]
            hh_end, mm_end, ss_end = parse_time_string(time_end)

            value_check=['on'] if row['Duration'].iloc[0] == 'Full Video' else []
            
            coord_outputs = row['Coordinates'].values[0]
            lat_str_l, lon_str_l = coord_outputs.split(",")
            lat, lon = float(lat_str_l), float(lon_str_l)
            marker = dl.Marker(position=[lat, lon],
                            children=[dl.Popup(coord_outputs)],
                            id='city-mark')
            viewport = {'center': [lat, lon], 'zoom': 14}
            
            return (
                False, dash.no_update, city_name_edit,
                row['Source'].values[0],
                row['Links'].values[0],
                row['Coordinates'].values[0],
                hh_st, mm_st, ss_st,
                hh_end, mm_end, ss_end,
                row['Time_of_the_day'].values[0],
                row['Weather'].values[0],
                row['Video_quality'].values[0],
                row['Camera_tilt'].values[0],
                row['Distance_from_building'].values[0],
                row['Occluded'].values[0],
                row['Terrain'].values[0],
                row['Logos_and_text'].values[0],
                row['Distortions'].values[0],
                row['Analyst'].values[0],
                row['Comments'].values[0] if row['Comments'].values[0] else "",
                row['Links'].values[0],False,False,value_check,dash.no_update,dash.no_update,viewport,[marker],dash.no_update,dash.no_update,dash.no_update
            )
      
        return (False, dash.no_update,"", "","","",0,0,0,0,0,0,"","","","","","","","","","","","",
                False,False,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)
   
    elif triggered_id == 'update.n_clicks':
        last_ver = ver_ops[-1]
        print(last_ver,video_version)
        try:
            if video_version  == last_ver:
                return (False, dash.no_update,stored_videoid) + (dash.no_update,) * 21 + (True,False,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)
            else:
                raise ValueError("Make sure you edit the latest version!")  
        except ValueError as e:              
            error_message = html.Div(
                [
                    html.H5("⚠️ Validation Error", style={"color": "red"}),
                    html.P(str(e), style={"color": "black"})
                ]
            )
            return(True, error_message,stored_videoid) + (dash.no_update,) * 21 + (False,False,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,)
    
    
    elif triggered_id =='confirm-yes.n_clicks' :
        try:
            general_validations_update = general_validations (analyst_edit,city_name_edit,distance_edit,occlusion_edit,
            terrain_edit,logos_edit,duration_edit, tod_edit,weather_edit,vq_edit,tilt_edit,sourceedit)
            if linkedit != df_city_edit[df_city_edit['Cut_ID'] == video_ver]["Links"].values[0]:
                valid_url_update = is_valid_url(linkedit)   
            else:
                valid_url_update = linkedit     
                    
            valid_coordinates_update = valid_coords(coord_edit)
            valid_duration_update = valid_dur(duration_edit)

            if not checkbox_ed:
                hours_st = minute_st_edit // 60
                minutes_st = minute_st_edit % 60
                start_time_edit = f"{hours_st:02}:{minutes_st:02}:{sec_st_edit:02}"
                hours = min_end_edit // 60
                minutes = min_end_edit % 60
                end_time_edit = f"{hours:02}:{minutes:02}:{sec_end_edit:02}"
                h, m, s = map(int, duration_edit.split(":"))
                dur_time =  f"{h:02}:{m:02}:{s:02}"
            else:
                start_time_edit =f"0:00"
                end_time_edit=f"0:00"
                dur_time= duration_edit

            df_city_edit['base'] = df_city_edit['Cut_ID'].str.extract(r'^(.*)_v\d+$')[0]
            other_rows = df_city_edit[df_city_edit['base'] != video_cut]

            # Check if the new URL and Coordinates pair exists elsewhere
            duplicate_match = other_rows[
                (other_rows['Links'] == valid_url_update)]

            if not duplicate_match.empty:
                if valid_coordinates_update in duplicate_match['Coordinates'].values:
                    raise ValueError("Video link and Coordinates already exist in another entry!")
                
                video_name_edit = df_city_edit[
                    df_city_edit['Links'] == valid_url_update
                ]['Title'].values[0]
                duplicate_match = duplicate_match.copy()  # Safely modify

                time_to_check_ed = parse_time(start_time_edit)
                duplicate_match['Start Time Parsed'] = duplicate_match['Start_Time'].apply(parse_time)
                duplicate_match['Finish Time Parsed'] = duplicate_match['Finish_Time'].apply(parse_time)

                dur_dup = duplicate_match.apply(
                    lambda row: row['Start Time Parsed'] <= time_to_check_ed <= row['Finish Time Parsed'],
                    axis=1
                )

                full_video_cross = duplicate_match.apply(
                    lambda row: row['Duration'] == "Full Video" ,
                    axis=1
                )
                if dur_dup.any() or full_video_cross.any():
                    raise ValueError("There's already this video with another crossing timing, please select another duration!")
                
            else:                
                if valid_url_update not in df_city_edit['Links'].values:  
                    if 'youtube' in valid_url_update and 'Youtube' in sourceedit:
                        with yt_dlp.YoutubeDL() as ydl:
                            info_dict = ydl.extract_info(valid_url_update, download=False)
                            video_name_edit = info_dict.get('title', 'No title found')
                    elif 'tiktok' in valid_url_update and 'Tiktok' in sourceedit:
                        with yt_dlp.YoutubeDL() as ydl:
                            info_dict = ydl.extract_info(valid_url_update, download=False)
                            video_name_edit = info_dict.get('title', 'No title found')
                            
                    elif 'facebook' in valid_url_update and 'facebook' in sourceedit:
                        with yt_dlp.YoutubeDL() as ydl:
                            info_dict = ydl.extract_info(valid_url_update, download=False)
                            video_name_edit = info_dict.get('title', 'No title found')
                    else:
                        raise ValueError("Video title not found - maybe not a matching source?")  
                    
                video_name_edit = df_city_edit[
                    df_city_edit['Links'] == valid_url_update]['Title'].values[0]
   
            video_og_dur = df_city_edit[
                    df_city_edit['Links'] == valid_url_update
                ]['Original_Duration'].values[0]
            
            updated_rec_id = df_city_edit[
                    df_city_edit['Links'] == valid_url_update
                ]['record_id'].values[0]
            
            updated_country = df_city_edit[
                    df_city_edit['Links'] == valid_url_update
                ]['Country'].values[0] 
            
            import datetime
            time_d=(datetime.datetime.now())
            formatted_datetime = time_d.strftime("%Y-%m-%d %H:%M:%S")                 
            
            match = re.match(r'^(.*)_v\d+$', video_ver)

            if match:
                cleaned_cut = match.group(1)
            else:
                cleaned_cut = video_ver  # fallback if no match
                 
            df_city_edit['base'] = df_city_edit['Cut_ID'].str.extract(r'^(.*)_v\d+$')[0]
            filt_clean = df_city_edit[df_city_edit['base'] == cleaned_cut]
            new_v = filt_clean.shape[0] + 1
            new_cut_v = f"{cleaned_cut}_v{new_v}"

            
            selected_inputs =[video_ver,updated_rec_id,updated_country,city_val, valid_url_update,video_name_edit,
            video_og_dur,sourceedit,valid_coordinates_update,start_time_edit,end_time_edit,
            dur_time,analyst_edit,tod_edit,terrain_edit,weather_edit,vq_edit,tilt_edit,distance_edit,
            occlusion_edit,distortions_edit,logos_edit,comments_edit]
            
            cut_ver_df = df_city_edit[df_city_edit['Cut_ID'] == video_ver].copy()
            
            mask = cut_ver_df.apply(lambda row: all(val in row.values for val in selected_inputs), axis=1)
            if mask.iloc[0]:
                raise ValueError ("Please change values to proceed!")
            latest_index = int(df_city_edit['Index'].iloc[-1]) + 1
            values_update= [latest_index,new_cut_v,updated_rec_id,updated_country,city_val, valid_url_update,video_name_edit,
            valid_coordinates_update,analyst_edit,sourceedit,video_og_dur,start_time_edit,end_time_edit,
            dur_time,tod_edit,terrain_edit,weather_edit,vq_edit,tilt_edit,distance_edit,
            occlusion_edit,distortions_edit,logos_edit,comments_edit,formatted_datetime]  

            
            try:              
                
                append_row_to_bq(values_update, table_id="airis-rnd-machines.Sample_Data.Geo_App_DB")
                # --- insert the new version into geo_cut ---

                latest_cut_index = int(cut_db_ed['Index'].iloc[-1]) + 1

                og_dured = df_city_edit[df_city_edit['Cut_ID'] == video_ver]['Original_Duration'].values[0]
                if not og_dured and not og_dur_val:
                    raise ValueError("Please insert original video duration")
                

                
                if not og_dured and og_dur_val:
                    og_dured = og_dur_val
                # Parse all the components as ints
                components = list(map(int, og_dured.split(":")))

                # Compute total seconds
                if len(components) == 3:
                    hours, mins, secs = components
                elif len(components) == 2:
                    mins, secs = components
                    hours = 0
                else:
                    raise ValueError(
                        "Original duration must be in H:MM:SS or MM:SS format"
                    )

                total_seconds = hours*3600 + mins*60 + secs

                # Re‐format as M:SS
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                formatted_dured = f"{hours:02}:{minutes:02}:{seconds:02d}"

                geo_cut_row = [
                    latest_cut_index,        # Index
                    new_cut_v,               # Cut_ID
                    updated_country,         # Country
                    city_val,                # City
                    valid_url_update,        # Links
                    video_name_edit,         # Title
                    'TBD',                   # Annotated File Name
                    start_time_edit,         # Cut_Start
                    end_time_edit,           # Cut_Finish
                    dur_time,                # Cut_Duration
                    'TBD',                   # Cut_Size
                    'TBD',                   # GCP_Bucket_URL
                    'FALSE',                 # Ignored (new version stays active)
                    analyst_edit,            # Validated_By
                    None,      # Upload_Time
                    sized,              # Video_Size_OG
                    formatted_dured                     # Video Duration_OG
                ]
                append_row_to_bq(geo_cut_row, table_id="airis-rnd-machines.Sample_Data.Cuts_DB")

                # --- mark all previous versions as ignored ---
                client = bigquery.Client()

                # First UPDATE: mark all previous versions as ignored
                query1 = """
                    UPDATE `airis-rnd-machines.Sample_Data.Cuts_DB`
                    SET Ignored = TRUE
                    WHERE Cut_ID != @new_cut_v
                    AND Cut_ID LIKE @like_pattern
                """

                job_config1 = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("new_cut_v", "STRING", new_cut_v),
                        bigquery.ScalarQueryParameter("like_pattern", "STRING", f"{cleaned_cut}_v%"),
                    ]
                )

                client.query(query1, job_config=job_config1).result()

                # Second UPDATE: set Validated_By for other versions
                query2 = """
                    UPDATE `airis-rnd-machines.Sample_Data.Cuts_DB`
                    SET Validated_By = @analyst_edit
                    WHERE Cut_ID LIKE @like_pattern
                    AND Cut_ID != @new_cut_v
                """

                job_config2 = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("analyst_edit", "STRING", analyst_edit),
                        bigquery.ScalarQueryParameter("like_pattern", "STRING", f"{cleaned_cut}_v%"),
                        bigquery.ScalarQueryParameter("new_cut_v", "STRING", new_cut_v),
                    ]
                )

                client.query(query2, job_config=job_config2).result()



            except Exception as e:
                error_window_update = html.Div([
                html.H5("⚠️ Update Failed", style={"color": "red"}),
                html.P(f"Could not update the database: {e}", style={"color": "black"})
             ])
                
            
                # Return the error window and leave everything else unchanged
                return (
                    True,                      # keep the update dialog open
                    error_window_update,       # show our new error
                    stored_videoid,            # keep current selection
                    *([dash.no_update] * 21),  # no other fields change
                    False,False,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update                     # disable Confirm button
                ,dash.no_update,dash.no_update,dash.no_update)
            
            #center= cities[cities['City Name'] == city_val]['CityCenter'].iloc[0]
            center = "41.8921503,12.4787812"
            lat, lon =map(float, center.split(",")) 
            map_center_city = {'center': [lat, lon], 'zoom': 10} 
                            
            result_window_update = html.Div([
                html.H1('Video Updated Successfully!'),
                html.Br(),
                html.H3("Video Details: "),
                html.Ul([
                    html.Li(f"City: {city_name_edit}"),
                    html.Li(f"Cut ID: {new_cut_v}"),
                    html.Li(f"Video Link: {valid_url_update}"),
                    html.Li(f"Video Name: {video_name_edit}"),
                    html.Li(f"Video Source: {sourceedit}"),                    
                    html.Li(f"Coordinates: {valid_coordinates_update}"),
                    html.Li(f"Start Time: {start_time_edit}"),
                    html.Li(f"Start Time: {end_time_edit}"),                    
                    html.Li(f"Video Duration: {valid_duration_update}"),
                    html.Li(f"Analyst: {analyst_edit}"),
                    html.Li(f"Time of the day: {tod_edit}"),
                    html.Li(f"Weather: {weather_edit}"), 
                    html.Li(f"Video Quality: {vq_edit}"), 
                    html.Li(f"Camera Tilt: {tilt_edit}"),
                    html.Li(f"Distance from a building: {distance_edit}"),
                    html.Li(f"Occlusion: {occlusion_edit}"),                                         
                    html.Li(f"Terrain: {terrain_edit}"),
                    html.Li(f"Logos and Text: {logos_edit}"),
                    html.Li(f"Distortions: {distortions_edit}"),
                    html.Li(f"Comments: {comments_edit}")                                                                    
                ])
            ])
            df1 = city_load_data("SELECT * FROM `airis-rnd-machines.Sample_Data.Geo_App_DB`")
            df2 = city_load_data("SELECT * FROM `airis-rnd-machines.Sample_Data.Cuts_DB`")
            checkbox_ed =[]  
            return (True,result_window_update,"", "","","",0,0,0,0,0,0,"","","",
                    "","","","","","","","","",False,False,checkbox_ed,
            dash.no_update,dash.no_update,map_center_city,[],"",df1.to_dict("records"),df2.to_dict("records"))      
        except ValueError as e:
            # If any validation fails, catch and show the error message
            error_message = html.Div(
                [
                    html.H5("⚠️ Validation Error", style={"color": "red"}),
                    html.P(str(e), style={"color": "black"})
                ]
            )
            return(True, error_message,stored_videoid) + (dash.no_update,) * 21 + (False,False,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,)
    
    elif triggered_id == 'delete.n_clicks':
        if stored_videoid:
            return (False, dash.no_update,stored_videoid) + (dash.no_update,) * 21 + (False,True,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,)
        else:
            raise ValueError("No  selected to remove!")
        
    elif triggered_id == 'delete_btn.n_clicks':
        if delete_password == "delete":
            try:

                
                center= cities[cities['City_Name'] == city_val]['CityCenter'].iloc[0]
                lat, lon =map(float, center.split(",")) 
                map_center_city = {'center': [lat, lon], 'zoom': 10} 
                remove_record_bq(video_ver,table_id="airis-rnd-machines.Sample_Data.Geo_App_DB")
                
                print(video_ver)
                    
                query1 = """
                    UPDATE `airis-rnd-machines.Sample_Data.Cuts_DB`
                    SET Ignored = TRUE
                    WHERE Cut_ID = @cut_id
                """

                # Define the query to set Validated_By
                query2 = """
                    UPDATE `airis-rnd-machines.Sample_Data.Cuts_DB`
                    SET Validated_By = @validated_by
                    WHERE Cut_ID = @cut_id
                """

                # Prepare parameters
                job_config1 = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("Cut_ID", "STRING", video_ver),
                    ]
                )

                job_config2 = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("Validated_By", "STRING", analyst_edit),
                        bigquery.ScalarQueryParameter("Cut_ID", "STRING", video_ver),
                    ]
                )

                # Run the update queries
                client.query(query1, job_config=job_config1).result()
                client.query(query2, job_config=job_config2).result()


                df1 = city_load_data("SELECT * FROM `airis-rnd-machines.Sample_Data.Geo_App_DB`")
                df2 = city_load_data("SELECT * FROM `airis-rnd-machines.Sample_Data.Cuts_DB`")
                
                df3 = df2[df2['Cut_ID'].str.contains(video_cut)].copy()
                if df3.shape[0] >= 2:
                    
                    cut_v = df3[df3['Ignored'] == 'true']['Cut_ID'].iloc[-1]

                    client = bigquery.Client()

                    # First UPDATE: Unmark "Ignored"
                    query1 = """
                        UPDATE `airis-rnd-machines.Sample_Data.Cuts_DB`
                        SET Ignored = FALSE
                        WHERE Cut_ID = @cut_v
                    """

                    job_config1 = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter("cut_v", "STRING", cut_v),
                        ]
                    )

                    client.query(query1, job_config=job_config1).result()

                    # Second UPDATE: Set "Validated_By"
                    query2 = """
                        UPDATE `airis-rnd-machines.Sample_Data.Cuts_DB`
                        SET Validated_By = @analyst_edit
                        WHERE Cut_ID = @cut_v
                    """

                    job_config2 = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter("analyst_edit", "STRING", analyst_edit),
                            bigquery.ScalarQueryParameter("cut_v", "STRING", cut_v),
                        ]
                    )

                    client.query(query2, job_config=job_config2).result()
                
                result_removal = f"{video_ver} has successfully removed !"

                return (True,result_removal,"Select a video id", "","","",0,0,0,0,0,0,"","","","","","","","","","","","",
            False,False,dash.no_update,dash.no_update,dash.no_update,map_center_city,[],"",df1.to_dict("records"),df2.to_dict("records"))   
            except Exception as e:
                # Build a Dash error window if the Sheets update fails
                error_window_update = html.Div([
                    html.H5("⚠️ Deletion Failed", style={"color": "red"}),
                    html.P(f"Could not update the sheet: {e}", style={"color": "black"})
                ])
                # Return the error window and leave everything else unchanged
                return (
                    True,                      # keep the update dialog open
                    error_window_update,       # show our new error
                    stored_videoid,            # keep current selection
                    *([dash.no_update] * 21),  # no other fields change
                    False,False,dash.no_update,dash.no_update,dash.no_update,
                    dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,                     # disable Confirm button
                )
            
      
        else:
            result_removal = "Incorrect Password, please try again!"
            return (True,result_removal,stored_videoid) + (dash.no_update,) * 21 + (False,False,dash.no_update,dash.no_update,dash.no_update,
            dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)        
      

    # No videoid selected, just update options
    return (False, dash.no_update,stored_videoid) + (dash.no_update,) * 21 + (False,False,dash.no_update,dash.no_update,dash.no_update,
    dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update)







 # Define the main layout with tabs
app.layout = html.Div(
    [
        dcc.Tabs(id='tabs', value='tab1', children=[
                dcc.Tab(
                    label='Geo-Tag Form',
                    children=insert_tab_layout(),
                    style=tab_style,
                    selected_style=selected_tab_style,
                    value='tab1'),
                dcc.Tab(
                    label='Geo-Tag Edit Mode',
                    children=edit_tab_layout(),
                    style=tab_style,
                    selected_style=selected_tab_style,
                    value='tab2'),
                

            ],
        ),
    ]
)                      
             
if __name__ == "__main__":
    app.run(host='100.118.47.56', port=8050, debug=True)