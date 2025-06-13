import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

# --- Set up Streamlit Page ---
st.set_page_config(layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .deckgl-container {
        height: 90vh !important; /* Increased map height to cover most of the page */
        margin-top: 0px !important; /* Remove space above map */
    }
    .main .block-container {
        padding-top: 1rem !important; /* Reduce space above heading and filters */
        padding-bottom: 1rem !important;
    }
    .summary-box, .filter-box {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 10px; /* Reduced margin for tighter layout */
        height: 100%;
    }
    .summary-box h3, .filter-box h3 {
        color: #2c3e50;
        margin-bottom: 10px;
        font-size: 10px;
    }
    .summary-box p, .filter-box p {
        margin: 5px 0;
        font-size: 15px;
        color: #34495e;
    }
    .summary-box .highlight, .filter-box .highlight {
        color: #e74c3c;
        font-weight: bold;
    }
    .summary-box .subheader, .filter-box .subheader {
        color: #2980b9;
        font-size: 18px;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .custom-table th, .custom-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }
    .custom-table th {
        background-color: #f8f9fa;
        color: #2c3e50;
    }
    .pydeck-tooltip-custom {
        background-color: #2c3e50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-family: sans-serif;
        border: 1px solid #e0e0e0;
        max-width: 300px;
        white-space: normal;
    }
    .pydeck-tooltip-custom div {
        margin-bottom: 2px;
    }
    .pydeck-tooltip-custom .highlight {
        color: #e74c3c;
        font-weight: light;
    }
    .sidebar .sidebar-content {
        width: 300px !important;
    }
    /* Ensure summary sections appear below map */
    .summary-section {
        margin-top: 20px;
    }
    /* Reduce size of filter boxes above the map (State, City, ZIP) */
    .stSelectbox > div > div > select {
        font-size: 10px !important; /* Smaller font for selectboxes */
        padding: 2px !important; /* Minimal padding */
        height: 24px !important; /* Reduced height */
        line-height: 1.2 !important; /* Adjust line height for better text alignment */
    }
    .stSelectbox > div > label {
        font-size: 10px !important; /* Smaller label font size */
        margin-bottom: 2px !important; /* Reduced margin */
    }
    /* Reduce column spacing for filters */
    .st-emotion-cache-1r4qj8v {
        padding: 0 5px !important; /* Reduced padding between filter columns */
    }
    </style>
""", unsafe_allow_html=True)

# --- OPTIMIZATION 1: EFFICIENT DATA LOADING & PREPARATION ---
@st.cache_data
def load_and_prepare_data():
    """
    Loads data and performs all initial, one-time transformations.
    This function is cached, so these expensive operations run only once.
    """
    try:
        # Update the path to your actual CSV file
        df = pd.read_csv(r"C:\Users\pankaj.kumar\Downloads\Main DB_1.csv")
    except FileNotFoundError:
        st.error("Default CSV file not found. Please upload the file to proceed.")
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.stop()

    # --- Initial Cleaning and Renaming ---
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    
    df.columns = [col.replace(' ', '_').lower() for col in df.columns]

    # --- Column Validation ---
    required_columns = [
        'npi_1', 'zip1', 'lat1', 'long1', 'city1', 'state1', 'influence_score_1', 'no._of_connections_hcp_1',
        'npi_2', 'zip2', 'lat2', 'long2', 'city2', 'state2', 'influence_score_2', 'no._of_connections_hcp_2',
        'maximum_influence_score', 'maximum_connections', 'npi_with_max_connections', 'papers', 'panels',
        'trials', 'affiliations', 'events', 'overall_connection_strength', 'interzip_connection',
        'intercity_connection', 'interstate_connection'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Your CSV is missing required columns (after renaming): {', '.join(missing_cols)}")
        st.stop()

    # --- Data Validation ---
    df = df.dropna(subset=['lat1', 'long1', 'lat2', 'long2', 'overall_connection_strength'])
    df = df[df['overall_connection_strength'] > 0]  # Ensure valid connections
    df['maximum_influence_score'] = pd.to_numeric(df['maximum_influence_score'], errors='coerce')
    df = df.dropna(subset=['maximum_influence_score'])

    # --- One-Time Data Structuring (Source/Target determination) ---
    df['is_source_1'] = df['influence_score_1'] >= df['influence_score_2']
    df['source_npi'] = np.where(df['is_source_1'], df['npi_1'], df['npi_2'])
    df['target_npi'] = np.where(df['is_source_1'], df['npi_2'], df['npi_1'])
    df['source_city'] = np.where(df['is_source_1'], df['city1'], df['city2'])
    df['source_state'] = np.where(df['is_source_1'], df['state1'], df['state2'])
    df['source_influence'] = np.where(df['is_source_1'], df['influence_score_1'], df['influence_score_2'])
    df['target_city'] = np.where(df['is_source_1'], df['city2'], df['city1'])
    df['target_state'] = np.where(df['is_source_1'], df['state2'], df['state1'])
    df['target_influence'] = np.where(df['is_source_1'], df['influence_score_2'], df['influence_score_1'])

    source_long = np.where(df['is_source_1'], df['long1'], df['long2'])
    source_lat = np.where(df['is_source_1'], df['lat1'], df['lat2'])
    target_long = np.where(df['is_source_1'], df['long2'], df['long1'])
    target_lat = np.where(df['is_source_1'], df['lat2'], df['lat1'])
    df['source_position'] = list(zip(source_long, source_lat))
    df['target_position'] = list(zip(target_long, target_lat))

    # --- One-Time Thickness Level Calculation ---
    def calculate_thickness_level(value):
        if 0 <= value <= 0.25:
            return 0.5
        elif 0.25 < value <= 0.5:
            return 1
        elif 0.5 < value <= 0.75:
            return 1.5
        elif 0.75 < value <= 1.00:
            return 2
        elif 1.00 < value <= 1.25:
            return 2.5
        elif 1.25 < value <= 1.5:
            return 3
        return 3.5  # Default for values above 1.5

    df['thickness_level'] = df['overall_connection_strength'].apply(calculate_thickness_level)
    
    return df

# --- Show loading spinner while preparing data ---
with st.spinner("Loading and preparing data..."):
    df = load_and_prepare_data()

# --- Cached Functions for Dynamic Filters ---
@st.cache_data
def get_cities_for_state(_df, state):
    if state == "All States":
        return sorted(list(set(_df['city1']).union(set(_df['city2']))))
    return sorted(list(set(_df[_df['state1'] == state]['city1']).union(set(_df[_df['state2'] == state]['city2']))))

@st.cache_data
def get_zips_for_city(_df, city):
    if city == "All Cities":
        return sorted(list(set(_df['zip1']).union(set(_df['zip2']))))
    return sorted(list(set(_df[_df['city1'] == city]['zip1']).union(set(_df[_df['city2'] == city]['zip2']))))

# --- Sidebar for Filters ---
with st.sidebar:
    st.markdown("<h3>Filters</h3>", unsafe_allow_html=True)

    # Double-Sided Sliders
    overall_strength_range = st.slider(
        "Overall Connection Strength Range",
        min_value=float(df['overall_connection_strength'].min()),
        max_value=float(df['overall_connection_strength'].max()),
        value=(float(df['overall_connection_strength'].min()), float(df['overall_connection_strength'].max())),
        step=0.1
    )

    max_influence_range = st.slider(
        "Influence Score Range",
        min_value=float(df['maximum_influence_score'].min()),
        max_value=float(df['maximum_influence_score'].max()),
        value=(float(df['maximum_influence_score'].min()), float(df['maximum_influence_score'].max())),
        step=0.1
    )

    max_connections_range = st.slider(
        "No. of Connections Range",
        min_value=int(df['maximum_connections'].min()),
        max_value=int(df['maximum_connections'].max()),
        value=(int(df['maximum_connections'].min()), int(df['maximum_connections'].max())),
        step=1
    )

    papers_range = st.slider(
        "Papers Published Range",
        min_value=int(df['papers'].min()),
        max_value=int(df['papers'].max()),
        value=(int(df['papers'].min()), int(df['papers'].max())),
        step=1
    )

    panels_range = st.slider(
        "Panels Range",
        min_value=int(df['panels'].min()),
        max_value=int(df['panels'].max()),
        value=(int(df['panels'].min()), int(df['panels'].max())),
        step=1
    )

    trials_range = st.slider(
        "Trials Range",
        min_value=int(df['trials'].min()),
        max_value=int(df['trials'].max()),
        value=(int(df['trials'].min()), int(df['trials'].max())),
        step=1
    )

    affiliations_range = st.slider(
        "Affiliations Range",
        min_value=int(df['affiliations'].min()),
        max_value=int(df['affiliations'].max()),
        value=(int(df['affiliations'].min()), int(df['affiliations'].max())),
        step=1
    )

    events_range = st.slider(
        "Promotional Events Range",
        min_value=int(df['events'].min()),
        max_value=int(df['events'].max()),
        value=(int(df['events'].min()), int(df['events'].max())),
        step=1
    )

    # Dropdowns
    interzip_options = ["All"] + sorted(list(df['interzip_connection'].dropna().unique().astype(str)))
    interzip = st.selectbox("Interzip Connection", interzip_options)

    intercity_options = ["All"] + sorted(list(df['intercity_connection'].dropna().unique().astype(str)))
    intercity = st.selectbox("Intercity Connection", intercity_options)

    interstate_options = ["All"] + sorted(list(df['interstate_connection'].dropna().unique().astype(str)))
    interstate = st.selectbox("Interstate Connection", interstate_options)

# --- Initialize Session State for Filters ---
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = "All States"
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "All Cities"
if 'selected_zip' not in st.session_state:
    st.session_state.selected_zip = "All ZIPs"

# --- Filter Row at Top ---
st.markdown("<h1 style='text-align: center; font-size: 24px; margin-top: 5; margin-bottom: 0px;'>HCP Network Map</h1>", unsafe_allow_html=True)
filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    states = ["All States"] + sorted(list(set(df['state1']).union(set(df['state2']))))
    st.session_state.selected_state = st.selectbox("State", states, key="state_filter")

cities = get_cities_for_state(df, st.session_state.selected_state)
if st.session_state.selected_city not in cities and st.session_state.selected_city != "All Cities":
    st.session_state.selected_city = "All Cities"

zips = get_zips_for_city(df, st.session_state.selected_city)
if st.session_state.selected_zip not in zips and st.session_state.selected_zip != "All ZIPs":
    st.session_state.selected_zip = "All ZIPs"

with filter_col2:
    st.session_state.selected_city = st.selectbox("City", ["All Cities"] + cities, key="city_filter")

with filter_col3:
    st.session_state.selected_zip = st.selectbox("ZIP", ["All ZIPs"] + zips, key="zip_filter")

# --- OPTIMIZATION 2: VECTORIZED & EFFICIENT FILTERING ---
with st.spinner("Applying filters..."):
    conditions = []
    
    # Top filter row
    if st.session_state.selected_state != "All States":
        conditions.append((df['state1'] == st.session_state.selected_state) | (df['state2'] == st.session_state.selected_state))
    
    if st.session_state.selected_city != "All Cities":
        conditions.append((df['city1'] == st.session_state.selected_city) | (df['city2'] == st.session_state.selected_city))

    if st.session_state.selected_zip != "All ZIPs":
        conditions.append((df['zip1'].astype(str) == str(st.session_state.selected_zip)) | (df['zip2'].astype(str) == str(st.session_state.selected_zip)))

    # Sidebar sliders
    conditions.append((df['overall_connection_strength'] >= overall_strength_range[0]) & (df['overall_connection_strength'] <= overall_strength_range[1]))
    conditions.append((df['maximum_influence_score'] >= max_influence_range[0]) & (df['maximum_influence_score'] <= max_influence_range[1]))
    conditions.append((df['maximum_connections'] >= max_connections_range[0]) & (df['maximum_connections'] <= max_connections_range[1]))
    conditions.append((df['papers'] >= papers_range[0]) & (df['papers'] <= papers_range[1]))
    conditions.append((df['panels'] >= panels_range[0]) & (df['panels'] <= panels_range[1]))
    conditions.append((df['trials'] >= trials_range[0]) & (df['trials'] <= trials_range[1]))
    conditions.append((df['affiliations'] >= affiliations_range[0]) & (df['affiliations'] <= affiliations_range[1]))
    conditions.append((df['events'] >= events_range[0]) & (df['events'] <= events_range[1]))

    # Sidebar dropdowns
    if interzip != "All":
        conditions.append(df['interzip_connection'].astype(str) == interzip)
    if intercity != "All":
        conditions.append(df['intercity_connection'].astype(str) == intercity)
    if interstate != "All":
        conditions.append(df['interstate_connection'].astype(str) == interstate)

    # Apply all filters at once
    if conditions:
        final_mask = np.logical_and.reduce(conditions)
        filtered_df = df[final_mask]
    else:
        filtered_df = df

# --- OPTIMIZATION 3: VECTORIZED TOOLTIP CREATION ---
if not filtered_df.empty:
    filtered_df['tooltip_html_content'] = (
        '<div class="pydeck-tooltip-custom">'
        f"<div><b>Source NPI:</b> " + filtered_df['source_npi'].astype(str) + "<br>"
        f"<b>City:</b> " + filtered_df['source_city'].astype(str) + "<br>"
        f"<b>State:</b> " + filtered_df['source_state'].astype(str) + "</div>"
        f"<div><b>Target NPI:</b> " + filtered_df['target_npi'].astype(str) + "<br>"
        f"<b>City:</b> " + filtered_df['target_city'].astype(str) + "<br>"
        f"<b>State:</b> " + filtered_df['target_state'].astype(str) + "</div>"
        '<div><b>Overall Connection Strength:</b> <span class="highlight">' + filtered_df['overall_connection_strength'].round(2).astype(str) + '</span></div>'
        '<div><b>Maximum Influence Score:</b> <span class="highlight">' + filtered_df['maximum_influence_score'].round(2).astype(str) + '</span></div>'
        '<div><b>No. of Connections:</b> <span class="highlight">' + filtered_df['maximum_connections'].round(2).astype(str) + '</span></div>'
        '<div><b>Papers:</b> ' + filtered_df['papers'].astype(str) + '</div>'
        '<div><b>Panels:</b> ' + filtered_df['panels'].astype(str) + '</div>'
        '<div><b>Trials:</b> ' + filtered_df['trials'].astype(str) + '</div>'
        '<div><b>Affiliations:</b> ' + filtered_df['affiliations'].astype(str) + '</div>'
        '<div><b>Events:</b> ' + filtered_df['events'].astype(str) + '</div>'
        '</div>'
    )
else:
    st.warning("No data available for the selected filters.")

# --- OPTIMIZATION 4: CLEANER MAP VIEWSTATE LOGIC ---
center_lat, center_lon, zoom_level = 39.8283, -98.5795, 3  # Default USA view

if not filtered_df.empty:
    if st.session_state.selected_zip != "All ZIPs":
        zoom_level = 10
    elif st.session_state.selected_city != "All Cities":
        zoom_level = 8
    elif st.session_state.selected_state != "All States":
        zoom_level = 5
    
    # Center map on the mean coordinates of the filtered data
    center_lat = filtered_df['lat1'].mean()
    center_lon = filtered_df['long1'].mean()

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=zoom_level,
    pitch=45
)

# --- Pydeck Layer and Tooltip Configuration ---
tooltip_config = {"html": "{tooltip_html_content}"}

line_layer = pdk.Layer(
    "ArcLayer",
    data=filtered_df,
    get_source_position="source_position",
    get_target_position="target_position",
    get_source_color=[255, 0, 0, 255],
    get_target_color=[255, 255, 0, 255],
    get_width="thickness_level",
    width_min_pixels=1,
    width_scale=2,
    pickable=True,
    auto_highlight=True,
    highlight_color=[255, 255, 0, 255]
)

# --- Render Map ---
deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/streets-v11",
    initial_view_state=view_state,
    layers=[line_layer],
    tooltip=tooltip_config,
    map_provider="mapbox",
    api_keys={"mapbox": "pk.eyJ1IjoicGFua2FqMjYyIiwiYSI6ImNtYnRieHNrejAxd24ybHM2ZmNuYmgycHEifQ.x953HmgosIBz3j4T47xNew"}  # Replace with your Mapbox API key
)

st.pydeck_chart(deck, use_container_width=True)

# --- Summary Section (Single Column Below Map) ---
st.markdown("<div class='summary-section'>", unsafe_allow_html=True)
st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
st.markdown("<h3>Network Summary</h3>", unsafe_allow_html=True)
if not filtered_df.empty and pd.api.types.is_numeric_dtype(filtered_df['maximum_influence_score']):
    total_connections = len(filtered_df)
    highest_influence = round(float(filtered_df['maximum_influence_score'].max()), 2)
    highest_connections = int(filtered_df['maximum_connections'].max())
    npi_max_connections = filtered_df['npi_with_max_connections'].mode().iloc[0]
    
    st.markdown(f"<p><b>Total Connections:</b> <span class='highlight'>{total_connections}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Highest Influence Score:</b> <span class='highlight'>{highest_influence}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Highest No. of Connections:</b> <span class='highlight'>{highest_connections}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p><b>NPI with Max Connections:</b> <span class='highlight'>{npi_max_connections}</span></p>", unsafe_allow_html=True)
else:
    st.markdown("<p>No data available for the selected filters or non-numeric influence scores.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)