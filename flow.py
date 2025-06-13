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
        height: 80vh !important;
    }
    .summary-box, .filter-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 10px;
        height: 100%;
    }
    .summary-box h3, .filter-box h3 {
        color: #2c3e50;
        margin-bottom: 15px;
        font-size: 20px;
    }
    .summary-box p, .filter-box p {
        margin: 5px 0;
        font-size: 16px;
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
    /* Updated tooltip class to be displayed by pydeck */
    .pydeck-tooltip-custom { /* Unique class for pydeck's managed tooltip */
        background-color: #2c3e50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-family: sans-serif;
        border: 1px solid #e0e0e0;
        max-width: 300px; /* Limit tooltip width */
        white-space: normal; /* Allow text wrapping */
    }
    .pydeck-tooltip-custom div {
        margin-bottom: 5px;
    }
    .pydeck-tooltip-custom .highlight {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading & Preparation ---
@st.cache_data
def load_data():
    """Loads data from a default path and strips whitespace."""
    try:
        df = pd.read_csv(r"C:\Users\pankaj.kumar\Downloads\Main DB_1.csv")
    except FileNotFoundError:
        st.error("Default CSV file not found at 'C:\\Users\\pankaj.kumar\\Downloads\\Main DB_1.csv'. Please upload the file or check the path.")
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.stop() # Stop if no file is found or uploaded

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
            
    # --- IMPORTANT: Rename columns to be compatible with pydeck tooltip ---
    # Replace spaces with underscores and convert to lowercase for all columns
    df.columns = [col.replace(' ', '_').lower() for col in df.columns]
    
    return df

# --- Show loading spinner ---
with st.spinner("Loading data..."):
    df = load_data()

# --- Column Validation (adjusting to new column names) ---
# Update required_columns to match the new snake_case, lowercase format
required_columns = [
    'npi_1', 'zip1', 'lat1', 'long1', 'city1', 'state1', 'influence_score_1',
    'npi_2', 'zip2', 'lat2', 'long2', 'city2', 'state2', 'influence_score_2',
    'pm_id_score', 'panel_score', 'trial_score', 'affiliation_score', 'events_score', 'overall_connection_strength'
]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Your CSV is missing the following required columns (after internal renaming): {', '.join(missing_cols)}")
    st.stop()

# --- Add No. of Connections ---
def calculate_connections(df_in):
    """Calculate number of connections per NPI efficiently."""
    npi_connections = pd.concat([
        df_in[['npi_1']].assign(connections=1).rename(columns={'npi_1': 'npi'}),
        df_in[['npi_2']].assign(connections=1).rename(columns={'npi_2': 'npi'})
    ])
    return npi_connections.groupby('npi')['connections'].sum().reset_index()

# --- Add new columns for source and target based on influence scores ---
df['is_source_1'] = df['influence_score_1'] >= df['influence_score_2']
df['source_npi'] = np.where(df['is_source_1'], df['npi_1'], df['npi_2'])
df['target_npi'] = np.where(df['is_source_1'], df['npi_2'], df['npi_1'])
df['source_city'] = np.where(df['is_source_1'], df['city1'], df['city2'])
df['source_state'] = np.where(df['is_source_1'], df['state1'], df['state2'])
df['source_influence'] = np.where(df['is_source_1'], df['influence_score_1'], df['influence_score_2'])
df['target_city'] = np.where(df['is_source_1'], df['city2'], df['city1'])
df['target_state'] = np.where(df['is_source_1'], df['state2'], df['state1'])
df['target_influence'] = np.where(df['is_source_1'], df['influence_score_2'], df['influence_score_1'])

# For positions (using renamed Lat/Long columns)
source_Long = np.where(df['is_source_1'], df['long1'], df['long2'])
source_Lat = np.where(df['is_source_1'], df['lat1'], df['lat2'])
target_Long = np.where(df['is_source_1'], df['long2'], df['long1'])
target_Lat = np.where(df['is_source_1'], df['lat2'], df['lat1'])
df['source_position'] = list(zip(source_Long, source_Lat))
df['target_position'] = list(zip(target_Long, target_Lat))

# --- Cached Functions for Filters (adjusting to new column names) ---
@st.cache_data
def get_cities_for_state(df, state):
    if state == "All States":
        return sorted(list(set(df['city1']).union(set(df['city2']))))
    else:
        return sorted(list(set(df[df['state1'] == state]['city1']).union(set(df[df['state2'] == state]['city2']))))

@st.cache_data
def get_zips_for_city(df, city):
    if city == "All Cities":
        return sorted(list(set(df['zip1']).union(set(df['zip2']))))
    else:
        return sorted(list(set(df[df['city1'] == city]['zip1']).union(set(df[df['city2'] == city]['zip2']))))

# --- Function to Get HCPs Dataframe (adjusting to new column names) ---
def get_hcps_df(data):
    hcp1 = data[['npi_1', 'city1', 'state1', 'lat1', 'long1', 'influence_score_1']].rename(columns={
        'npi_1': 'npi', 'city1': 'city', 'state1': 'state', 'lat1': 'lat', 'long1': 'long', 'influence_score_1': 'influence_score'
    })
    hcp2 = data[['npi_2', 'city2', 'state2', 'lat2', 'long2', 'influence_score_2']].rename(columns={
        'npi_2': 'npi', 'city2': 'city', 'state2': 'state', 'lat2': 'lat', 'long2': 'long', 'influence_score_2': 'influence_score'
    })
    all_hcps = pd.concat([hcp1, hcp2], ignore_index=True)
    all_hcps = all_hcps.drop_duplicates(subset='npi', keep='first')
    return all_hcps

# --- Initialize Session State for Filters ---
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = "All States"
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "All Cities"
if 'selected_zip' not in st.session_state:
    st.session_state.selected_zip = "All ZIPs"
if 'selected_score' not in st.session_state:
    st.session_state.selected_score = "All"
if 'selected_criterion' not in st.session_state:
    st.session_state.selected_criterion = "Overall Score"

# --- Filter Row at Top ---
st.markdown("<h1 style='text-align: center; font-size: 24px;'>HCP Connections Map</h1>", unsafe_allow_html=True)
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 2, 2])

with filter_col1:
    states = ["All States"] + sorted(list(set(df['state1']).union(set(df['state2']))))
    st.session_state.selected_state = st.selectbox("State", states, index=states.index(st.session_state.selected_state), key="state_filter")

with filter_col2:
    cities = get_cities_for_state(df, st.session_state.selected_state)
    # Ensure current selected_city is valid for the new state, reset if not
    if st.session_state.selected_city not in cities and st.session_state.selected_city != "All Cities":
        st.session_state.selected_city = "All Cities"
    st.session_state.selected_city = st.selectbox("City", ["All Cities"] + cities, 
                                                  index=(cities.index(st.session_state.selected_city) + 1 if st.session_state.selected_city in cities else 0), 
                                                  key="city_filter")


with filter_col3:
    zips = get_zips_for_city(df, st.session_state.selected_city)
    # Ensure current selected_zip is valid for the new city, reset if not
    if st.session_state.selected_zip not in zips and st.session_state.selected_zip != "All ZIPs":
        st.session_state.selected_zip = "All ZIPs"
    st.session_state.selected_zip = st.selectbox("ZIP", ["All ZIPs"] + zips, 
                                                 index=(zips.index(st.session_state.selected_zip) + 1 if st.session_state.selected_zip in zips else 0), 
                                                 key="zip_filter")

with filter_col4:
    score_ranges = ["All", "Low (≤ 0.5)", "Medium (0.5 - 1.0)", "High (> 1.0)"]
    st.session_state.selected_score = st.selectbox("Connection Strength", score_ranges, index=score_ranges.index(st.session_state.selected_score), key="score_filter")

# --- Apply Filters (adjusting to new column names) ---
with st.spinner("Applying filters..."):
    filtered_df = df.copy()

    if st.session_state.selected_state != "All States":
        filtered_df = filtered_df[
            (filtered_df['state1'] == st.session_state.selected_state) | (filtered_df['state2'] == st.session_state.selected_state)
        ]

    if st.session_state.selected_city != "All Cities":
        filtered_df = filtered_df[
            (filtered_df['city1'] == st.session_state.selected_city) | (filtered_df['city2'] == st.session_state.selected_city)
        ]

    if st.session_state.selected_zip != "All ZIPs":
        filtered_df = filtered_df[
            (filtered_df['zip1'] == st.session_state.selected_zip) | (filtered_df['zip2'] == st.session_state.selected_zip)
        ]

    if st.session_state.selected_score != "All":
        if st.session_state.selected_score == "Low (≤ 0.5)":
            filtered_df = filtered_df[filtered_df['overall_connection_strength'] <= 0.5]
        elif st.session_state.selected_score == "Medium (0.5 - 1.0)":
            filtered_df = filtered_df[
                (filtered_df['overall_connection_strength'] > 0.5) & 
                (filtered_df['overall_connection_strength'] <= 1.0)
            ]
        elif st.session_state.selected_score == "High (> 1.0)":
            filtered_df = filtered_df[filtered_df['overall_connection_strength'] > 1.0]

# --- Add thickness_level column (adjusting to new column names) ---
filtered_df['thickness_level'] = filtered_df['overall_connection_strength'].apply(
    lambda x: 0.5 if x <= 0.5 else (1.5 if x <= 1.0 else 3)
)

# --- NEW: Create a pre-formatted HTML tooltip column ---
if not filtered_df.empty:
    filtered_df['tooltip_html_content'] = filtered_df.apply(lambda row: f"""
        <div class="pydeck-tooltip-custom">
            <div><b>Source NPI:</b> {row['source_npi']}<br>
                <b>City:</b> {row['source_city']}<br>
                <b>State:</b> {row['source_state']}</div>
            <div><b>Target NPI:</b> {row['target_npi']}<br>
                <b>City:</b> {row['target_city']}<br>
                <b>State:</b> {row['target_state']}</div>
            <div><b>Overall Connection Strength:</b> <span class="highlight">{row['overall_connection_strength']:.2f}</span></div>
        </div>
    """, axis=1)

# --- Prepare HCPs Data and Connections (adjusting to new column names) ---
if not filtered_df.empty:
    hcps_df = get_hcps_df(filtered_df)
    connections = calculate_connections(filtered_df)
    hcps_df = hcps_df.merge(connections, on='npi', how='left').fillna({'connections': 0})
    hcps_df['radius'] = (hcps_df['influence_score'] / hcps_df['influence_score'].max()) * 50
else:
    hcps_df = pd.DataFrame(columns=['npi', 'city', 'state', 'lat', 'long', 'influence_score', 'connections'])
    st.warning("No data available after applying filters.")

# --- Map View State with Zoom Logic (adjusting to new column names) ---
if 'current_zoom_level' not in st.session_state:
    st.session_state.current_zoom_level = 3

# Initialize with default values
center_lat = 39.8283
center_lon = -98.5795
zoom_level = 3

if not filtered_df.empty:
    if st.session_state.selected_zip != "All ZIPs":
        zoom_level = 10
        temp_df = filtered_df[(filtered_df['zip1'] == st.session_state.selected_zip) | (filtered_df['zip2'] == st.session_state.selected_zip)]
        if not temp_df.empty:
            center_lat = temp_df['lat1'].mean()
            center_lon = temp_df['long1'].mean()
    elif st.session_state.selected_city != "All Cities":
        zoom_level = 8
        temp_df = filtered_df[(filtered_df['city1'] == st.session_state.selected_city) | (filtered_df['city2'] == st.session_state.selected_city)]
        if not temp_df.empty:
            center_lat = temp_df['lat1'].mean()
            center_lon = temp_df['long1'].mean()
    elif st.session_state.selected_state != "All States":
        zoom_level = 5
        temp_df = filtered_df[(filtered_df['state1'] == st.session_state.selected_state) | (filtered_df['state2'] == st.session_state.selected_state)]
        if not temp_df.empty:
            center_lat = temp_df['lat1'].mean()
            center_lon = temp_df['long1'].mean()
    else: # If no specific filter is applied, use overall mean
        center_lat = filtered_df['lat1'].mean()
        center_lon = filtered_df['long1'].mean()

st.session_state.current_zoom_level = zoom_level

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=zoom_level,
    pitch=45,
    bearing=0
)

# --- Modified Tooltip Configuration for Pydeck ---
# Now, the tooltip doesn't need templating; it just takes the 'html' property directly.
# The 'html' property is derived from the 'tooltip_html_content' column.
tooltip_config = {
    "html": "{tooltip_html_content}"
}


# --- Individual Connection Layers (adjusting to new column names) ---
line_layer = pdk.Layer(
    "ArcLayer",
    data=filtered_df,
    get_source_position="source_position",
    get_target_position="target_position",
    get_source_color=[255, 0, 0, 255],  # Red
    get_target_color=[255, 255, 0, 255],  # Yellow
    get_width="thickness_level",
    width_min_pixels=1,
    width_scale=2,
    pickable=True,
    auto_highlight=True,
    highlight_color=[255, 255, 0, 255] # Yellow highlight on hover
)

node_layer = pdk.Layer(
    "ScatterplotLayer",
    data=hcps_df,
    get_position=["long", "lat"], # Using renamed columns
    get_fill_color=[255, 0, 0, 160],  # Red, semi-transparent
    get_radius="radius",
    radius_min_pixels=2,
    pickable=True
)

layers = [line_layer, node_layer]

# --- Render Map ---
deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/streets-v11",
    initial_view_state=view_state,
    layers=layers,
    tooltip=tooltip_config, # Use the new tooltip_config
    map_provider="mapbox",
    api_keys={"mapbox": "pk.eyJ1IjoicGFua2FqMjYyIiwiYSI6ImNtYnRieHNrejAxd24ybHM2ZmNuYmgycHEifQ.x953HmgosIBz3j4T47xNew"}
)

st.pydeck_chart(deck, use_container_width=True)

# --- Summary and Filter Sections (adjusting to new column names) ---
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
    st.markdown("<h3>Network Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"<p>Total Connections Shown: <span class='highlight'>{len(filtered_df):,}</span></p>", unsafe_allow_html=True)
    
    avg_strength = filtered_df['overall_connection_strength'].mean() if not filtered_df.empty else 0
    st.markdown(f"<p>Average Connection Strength: <span class='highlight'>{avg_strength:.2f}</span></p>", unsafe_allow_html=True)
    
    # --- CORRECTED LOGIC FOR UNIQUE STATES AND CITIES ---
    if st.session_state.selected_state != "All States":
        # If a state is selected, it's 1 unique state.
        unique_states_count = 1
        display_states = st.session_state.selected_state
    else:
        # Otherwise, count unique states in the filtered data (from both NPI1 and NPI2)
        unique_states_set = set(filtered_df['state1']).union(set(filtered_df['state2']))
        unique_states_count = len(unique_states_set)
        display_states = ", ".join(sorted(list(unique_states_set))) if unique_states_set else "N/A"

    if st.session_state.selected_city != "All Cities":
        # If a city is selected, it's 1 unique city.
        unique_cities_count = 1
        display_cities = st.session_state.selected_city
    else:
        # If a state is selected, but not a city, count cities within that state from filtered data
        if st.session_state.selected_state != "All States":
            unique_cities_set = set(filtered_df['city1']).union(set(filtered_df['city2']))
            # Further filter cities to only those belonging to the selected state (if any)
            # This is already handled by filtered_df if the state filter is applied, but good to be explicit
            unique_cities_set = unique_cities_set.intersection(set(get_cities_for_state(df, st.session_state.selected_state)))
            unique_cities_count = len(unique_cities_set)
            display_cities = ", ".join(sorted(list(unique_cities_set))) if unique_cities_set else "N/A"
        else:
            # If no state or city selected, count unique cities in the filtered data
            unique_cities_set = set(filtered_df['city1']).union(set(filtered_df['city2']))
            unique_cities_count = len(unique_cities_set)
            display_cities = ", ".join(sorted(list(unique_cities_set))) if unique_cities_set else "N/A"

    st.markdown(f"<p>Unique States: <span class='highlight'>{unique_states_count}</span></p>", unsafe_allow_html=True)

    st.markdown(f"<p>Unique Cities: <span class='highlight'>{unique_cities_count}</span></p>", unsafe_allow_html=True)
    # --- END OF CORRECTED LOGIC ---

    st.markdown("</div>", unsafe_allow_html=True) # Close the summary-box div

with right_col:
    st.markdown("<div class='filter-box'>", unsafe_allow_html=True)
    st.markdown("<h3>Top HCPs Filter</h3>", unsafe_allow_html=True)
    criterion = st.selectbox("Select Criterion", ["Overall Score", "Number of Connections"], key="criterion_filter")
    st.markdown("<hr style='border: 1px solid #e0e0e0; margin: 10px 0;'>", unsafe_allow_html=True)
    if not hcps_df.empty:
        if criterion == "Overall Score":
            top10 = hcps_df.sort_values(by='influence_score', ascending=False).head(10)
        else:
            top10 = hcps_df.sort_values(by='connections', ascending=False).head(10)
        st.dataframe(top10[['npi', 'city', 'state', 'influence_score', 'connections']], use_container_width=True)
    else:
        st.markdown("<p>No data available.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # Close the filter-box div