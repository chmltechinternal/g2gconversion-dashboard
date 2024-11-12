import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
from scipy.special import expit

# Define optimal ranges with sensitivity and base impacts
OPTIMAL_RANGES = {
    't2': {
        'min': 195, 'max': 205, 'optimal': 200,
        'weight': 0.27,
        'sensitivity': 0.217985,
        'base_impact': 30
    },
    't3': {
        'min': 195, 'max': 205, 'optimal': 200,
        'weight': 0.25,
        'sensitivity': 0.136957,
        'base_impact': 25
    },
    't4': {
        'min': 195, 'max': 205, 'optimal': 200,
        'weight': 0.40,
        'sensitivity': 0.181967,
        'base_impact': 25
    },
    't1': {
        'min': 190, 'max': 200, 'optimal': 195,
        'weight': 0.10,
        'sensitivity': 0.069769,
        'base_impact': 10
    },
    't5': {
        'min': 190, 'max': 200, 'optimal': 195,
        'weight': 0.05,
        'sensitivity': 0.003058,
        'base_impact': 10
    },
    'lhsv': {
        'min': 0.5, 'max': 0.7, 'optimal': 0.6,
        'weight': 0.05,
        'sensitivity': 0.016678,
        'base_impact': 5
    },
    'h2gly_ratio': {
        'min': 6, 'max': 7, 'optimal': 6.5,
        'weight': 0.05,
        'sensitivity': 0.004925,
        'base_impact': 5
    },
    'liquid_feed': {
        'min': 50, 'max': 150, 'optimal': 100,
        'weight': 0.05,
        'sensitivity': 0.000073,
        'base_impact': 5
    },
    'hydrogen_flow': {
        'min': 300, 'max': 600, 'optimal': 450,
        'weight': 0.05,
        'sensitivity': 0.000058,
        'base_impact': 5
    },
    'top_pressure': {
        'min': 20, 'max': 40, 'optimal': 30,
        'weight': 0.05,
        'sensitivity': 0.003176,
        'base_impact': 5
    },
    'bottom_pressure': {
        'min': 15, 'max': 35, 'optimal': 25,
        'weight': 0.05,
        'sensitivity': 0.001230,
        'base_impact': 5
    },
    'feed_ph': {
        'min': 6, 'max': 8, 'optimal': 7,
        'weight': 0.05,
        'sensitivity': 0.010348,
        'base_impact': 5
    }
}

def smooth_transition(x, center, sensitivity):
    """Create a smooth transition using sigmoid function"""
    return expit(-(x - center) * sensitivity)

def calculate_parameter_impact(value, param_info):
    """Calculate normalized impact with smooth transitions"""
    optimal = param_info['optimal']
    min_val = param_info['min']
    max_val = param_info['max']
    sensitivity = param_info['sensitivity']
    base_impact = param_info['base_impact']

    # Normalize distance from optimal
    range_size = max_val - min_val
    normalized_distance = abs(value - optimal) / range_size

    # Calculate base effect using smooth transition
    base_effect = smooth_transition(normalized_distance, 0, 1/sensitivity)

    # Scale effect by base impact
    impact = base_impact * base_effect

    # Apply small penalty for being outside optimal range
    if value < min_val or value > max_val:
        excess = min(abs(value - optimal) / range_size, 1)
        impact *= (1 - 0.2 * excess)

    return impact

def calculate_total_conversion(params):
    """Calculate glycerol conversion with normalized impacts"""
    # Calculate derived parameters
    pressure_diff = params['top_pressure'] - params['bottom_pressure']
    temps = [params['t1'], params['t2'], params['t3'], params['t4'], params['t5']]
    avg_temperature = sum(temps) / len(temps)
    temp_range = max(temps) - min(temps)

    # Calculate individual impacts
    impacts = {name: calculate_parameter_impact(value, OPTIMAL_RANGES[name])
              for name, value in params.items()}

    # Calculate base conversion with temperature average effect
    base_conversion = 60  # Minimum expected conversion

    # Sum weighted impacts
    total_impact = sum(impacts.values())

    # Calculate final conversion with smoother scaling
    conversion = base_conversion + total_impact

    # Ensure realistic bounds
    conversion = np.clip(conversion, 0, 100)

    return conversion, impacts



# Define your custom CSS
custom_css = """
<style>
.prediction-box {
 background-color: #f0f2f6;
 padding: 10px;
 border-radius: 5px;
 text-align: center;

}
</style>
"""


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Glycerin-to-Glycol Conversion dashboard',
    #page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()


def get_lrc_data():
    """Grab LRC data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME2 = Path(__file__).parent/'data/LRC_G1_Data_clean.csv'
    lrcg1_df = pd.read_csv(DATA_FILENAME2)

    # The data above has columns like:
    # - Sample ID
    # - Sample Number
    # - [Stuff I don't care about]
    # - T1 (°C)
    # - T2 (°C)
    # - T3 (°C)
    # - ...
    # - T5 (°C)
    #

    # Convert temp from string to integers
    lrcg1_df['T1 (°C)'] = pd.to_numeric(lrcg1_df['T1 (°C)'])
    lrcg1_df['T2 (°C)'] = pd.to_numeric(lrcg1_df['T2 (°C)'])
    lrcg1_df['T3 (°C)'] = pd.to_numeric(lrcg1_df['T3 (°C)'])
    lrcg1_df['T4 (°C)'] = pd.to_numeric(lrcg1_df['T4 (°C)'])
    lrcg1_df['T5 (°C)'] = pd.to_numeric(lrcg1_df['T5 (°C)'])

    return lrcg1_df

lrcg1_df = get_lrc_data()


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page. Commented out code below shows with an icon
# :earth_americas: Glycerin-to-Glycol Conversion dashboard

#title only
'''
# Glycerin-to-Glycol Conversion dashboard

Advanced model with normalized parameter impacts.
'''
#set the tab containers
tempControl, procParam, pressurePH = st.tabs(["Temperature Control", "Process Parameters", "Pressure & pH Control"])

with tempControl:
    min_value = 180
    max_value = 220
    range_T2 = st.slider(
    'T2 (°C) - Critical:',
    min_value=180,
    max_value=200,
    value=[min_value, max_value])
    
    range_T3 = st.slider(
    'T3 (°C):',
    min_value=180,
    max_value=200,
    value=[min_value, max_value])
    
    range_T4 = st.slider(
    'T4 (°C):',
    min_value=180,
    max_value=200,
    value=[min_value, max_value])
    
    range_T1 = st.slider(
    'T1 (°C):',
    min_value=180,
    max_value=195,
    value=[min_value, max_value])
    
    range_T5 = st.slider(
    'T5 (°C):',
    min_value=180,
    max_value=195,
    value=[min_value, max_value])


with procParam:
    range_LHSV = st.slider(
    'LHSV (1/h):',
    value=[0.00, 0.60])
    
    range_H2_GLY = st.slider(
    'H2:GLY Ratio:',
    value=[0.00, 6.50])
    
    range_LF = st.slider(
    'Liquid Feed (g/h):',
    value=[0.00, 100.00])
    
    range_HF = st.slider(
    'Hydrogen Flow (mL/min):',
    value=[0.00, 450.00])

with pressurePH:
    range_TP = st.slider(
    'Top Pressure (bar):',
    value=[0.00, 30.00])
    
    range_BP = st.slider(
    'Bottom Pressure (bar):',
    value=[0.00, 25.00])
    
    range_FpH = st.slider(
    'Feed pH:',
    value=[0.00, 7.00])
    
# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Use the custom class in a container
st.markdown('<div class="prediction-box"><h2 style="margin: 0; color: #2c3e50;">Predicted Glycerol Conversion</h2><p style="font-size: 2.5em; margin: 10px 0; color: #2980b9;">{conversion:.1f}%</p></div>', unsafe_allow_html=True)




# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )
