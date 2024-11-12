import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
    DATA_FILENAME2 = Path(__file__).parent/'data/LRCG1Dataclean.csv'
    lrcg1_df = pd.read_csv(DATA_FILENAME2)

    # The data above has columns like:
    # - Sample ID
    # - Sample Number
    # - ...
    # - T1 (°C)
    # - T2 (°C)
    # - T3 (°C)
    # - ...
    # - T5 (°C)
    #

    # Convert temp from string to float
    lrcg1_df['T1C'] = pd.to_numeric(lrcg1_df['T1C'])
    lrcg1_df['T2C'] = pd.to_numeric(lrcg1_df['T2C'])
    lrcg1_df['T3C'] = pd.to_numeric(lrcg1_df['T3C'])
    lrcg1_df['T4C'] = pd.to_numeric(lrcg1_df['T4C'])
    lrcg1_df['T5C'] = pd.to_numeric(lrcg1_df['T5C'])

    # Convert top pressure from integer to float
    lrcg1_df['TopReactorPressurepsi'] = pd.to_numeric(lrcg1_df['TopReactorPressurepsi']+0.0)
    lrcg1_df['BottomReactorPressurepsi'] = pd.to_numeric(lrcg1_df['BottomReactorPressurepsi']+0.0)

    return lrcg1_df

lrcg1_df = get_lrc_data()


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page. Commented out code below shows with an icon
# :earth_americas: Glycerin-to-Glycol Conversion dashboard

#title only
'''
# Glycerin-to-Glycol Conversion dashboard
'''
st.subheader('Glycerol Conversion Calculator', divider='red')
st.markdown("Advanced model with normalized parameter impacts")

col_sliders, col_prediction = st.columns([4,2])

with col_sliders:
        
# Create tabs
    tempControl, procParam, pressurePH = st.tabs(["Temperature Control", "Process Parameters", "Pressure & pH Control"])

with tempControl:
    #T2 slider
    min_value = lrcg1_df['T2C'].min()
    median_value = lrcg1_df['T2C'].median()
    max_value = lrcg1_df['T2C'].max()
    t2 = st.slider(
    'T2 (°C) - Critical:',
    min_value = lrcg1_df['T2C'].min(),
    max_value = lrcg1_df['T2C'].max(),
    value=median_value)
    
    #T3 slider
    min_value = lrcg1_df['T3C'].min()
    median_value = lrcg1_df['T3C'].median()
    max_value = lrcg1_df['T3C'].max()
    t3 = st.slider(
    'T3 (°C):',
    min_value = lrcg1_df['T3C'].min(),
    max_value = lrcg1_df['T3C'].max(),
    value=median_value)
     
    #T4 slider
    min_value = lrcg1_df['T4C'].min()
    median_value = lrcg1_df['T4C'].median()
    max_value = lrcg1_df['T4C'].max()
    t4 = st.slider(
    'T4 (°C):',
    min_value = lrcg1_df['T4C'].min(),
    max_value = lrcg1_df['T4C'].max(),
    value=median_value)
    
    #T1 slider
    min_value = lrcg1_df['T1C'].min()
    median_value = lrcg1_df['T1C'].median()
    max_value = lrcg1_df['T1C'].max()
    t1 = st.slider(
    'T1 (°C):',
    min_value = lrcg1_df['T1C'].min(),
    max_value = lrcg1_df['T1C'].max(),
    value=median_value)
    
    #T5 slider
    min_value = lrcg1_df['T5C'].min()
    median_value = lrcg1_df['T5C'].median()
    max_value = lrcg1_df['T5C'].max()
    t5 = st.slider(
    'T5 (°C):',
    min_value = lrcg1_df['T5C'].min(),
    max_value = lrcg1_df['T5C'].max(),
    value=median_value)


with procParam:
    
    #LHSV slider
    min_value = lrcg1_df['LHSV1husescatalystvolume'].min()
    median_value = lrcg1_df['LHSV1husescatalystvolume'].median()
    max_value = lrcg1_df['LHSV1husescatalystvolume'].max()
    lhsv = st.slider(
    'LHSV (1/h):',
    min_value = lrcg1_df['LHSV1husescatalystvolume'].min(),
    max_value = lrcg1_df['LHSV1husescatalystvolume'].max(),
    value=median_value)

    #H2:GLY slider
    min_value = lrcg1_df['H2GLYMolarRatio'].min()
    median_value = lrcg1_df['H2GLYMolarRatio'].median()
    max_value = lrcg1_df['H2GLYMolarRatio'].max()
    h2gly_ratio = st.slider(
    'H2:GLY Ratio:',
    min_value = lrcg1_df['H2GLYMolarRatio'].min(),
    max_value = lrcg1_df['H2GLYMolarRatio'].max(),
    value=median_value)

    #Liquid Feed slider
    min_value = lrcg1_df['LiquidFeedSetpointmLmin'].min()
    median_value = lrcg1_df['LiquidFeedSetpointmLmin'].median()
    max_value = lrcg1_df['LiquidFeedSetpointmLmin'].max()
    liquid_feed = st.slider(
    'Liquid Feed (g/h):',
    min_value = lrcg1_df['LiquidFeedSetpointmLmin'].min(),
    max_value = lrcg1_df['LiquidFeedSetpointmLmin'].max(),
    value=median_value)
    
    #Hydrogen Flow slider
    min_value = lrcg1_df['HydrogenGasFlowLmin'].min()
    median_value = lrcg1_df['HydrogenGasFlowLmin'].median()
    max_value = lrcg1_df['HydrogenGasFlowLmin'].max()
    hydrogen_flow = st.slider(
    'Hydrogen Flow (mL/min):',
    min_value = lrcg1_df['HydrogenGasFlowLmin'].min(),
    max_value = lrcg1_df['HydrogenGasFlowLmin'].max(),
    value=median_value)


with pressurePH:

    #Top Reactor Pressure slider
    min_value = lrcg1_df['TopReactorPressurepsi'].min()
    median_value = lrcg1_df['TopReactorPressurepsi'].median()
    max_value = lrcg1_df['TopReactorPressurepsi'].max()
    top_pressure = st.slider(
    'Top Pressure (bar):',
    min_value = lrcg1_df['TopReactorPressurepsi'].min(),
    max_value = lrcg1_df['TopReactorPressurepsi'].max(),
    value=median_value)

    #Bottom Reactor Pressure slider
    min_value = lrcg1_df['BottomReactorPressurepsi'].min()
    median_value = lrcg1_df['BottomReactorPressurepsi'].median()
    max_value = lrcg1_df['BottomReactorPressurepsi'].max()
    bottom_pressure = st.slider(
    'Bottom Pressure (bar):',
    min_value = lrcg1_df['BottomReactorPressurepsi'].min(),
    max_value = lrcg1_df['BottomReactorPressurepsi'].max(),
    value=median_value)
    
    #Feed pH slider
    min_value = lrcg1_df['FeedpH'].min()
    median_value = lrcg1_df['FeedpH'].median()
    max_value = lrcg1_df['FeedpH'].max()
    feed_ph = st.slider(
    'Feed pH:',
    min_value = lrcg1_df['FeedpH'].min(),
    max_value = lrcg1_df['FeedpH'].max(),
    value=median_value)

 # Calculate conversion and impacts
    params = {
        't2': t2,
        't3': t3,
        't4': t4,
        't1': t1,
        't5': t5,
        'lhsv': lhsv,
        'h2gly_ratio': h2gly_ratio,
        'liquid_feed': liquid_feed,
        'hydrogen_flow': hydrogen_flow,
        'top_pressure': top_pressure,
        'bottom_pressure': bottom_pressure,
        'feed_ph': feed_ph
    }

    conversion, impacts = calculate_total_conversion(params)

#Custom css
st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            text-align:left;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
        }
        .metric-container {
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 5px;
            border: 2px solid #2c3e50;
            text-align: center;
            margin: 20px 0;
        }
        .subheading{
            font-weight:bold;
            font-size: 22px;
            }

        .subsubheading {
            color: red;
            font-size: 18px;
        }
        .info-text {
            font-size: 14px;
            line-height:14px;
        }
        </style>
    """, unsafe_allow_html=True)


with col_prediction:
     
# Display prediction
    st.markdown(f"""
    <div class="metric-container">
        <h3 style="text-align:center">Predicted Glycerol Conversion</h3>
        <h1 style="text-align:center">{conversion:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)

 # Display parameter impacts
st.markdown("""<p class="subheading">Parameter Contributions</p>""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([3,3,2])

with col1:
        st.markdown("""<p class="subsubheading">Temperature Impacts</p>""", unsafe_allow_html=True)
        for temp in ['t2', 't3', 't4', 't1', 't5']:
            st.write(f"""<p class="info-text">{temp.upper()}: +{impacts[temp]:.1f}%</p>""", unsafe_allow_html=True)

with col2:
        st.markdown("""<p class="subsubheading">Process Parameters</p>""", unsafe_allow_html=True)
        st.write(f"""<p class="info-text">LHSV: +{impacts['lhsv']:.1f}%</p>""", unsafe_allow_html=True)
        st.write(f"""<p class="info-text">H2:GLY Ratio: +{impacts['h2gly_ratio']:.1f}%</p>""", unsafe_allow_html=True)
        st.write(f"""<p class="info-text">Liquid Feed: +{impacts['liquid_feed']:.1f}%</p>""", unsafe_allow_html=True)
        st.write(f"""<p class="info-text">Hydrogen Flow: +{impacts['hydrogen_flow']:.1f}%</p>""", unsafe_allow_html=True)

with col3:
        st.markdown("""<p class="subsubheading">Pressure & pH</p>""", unsafe_allow_html=True)
        st.write(f"""<p class="info-text">Top Pressure: +{impacts['top_pressure']:.1f}%</p>""", unsafe_allow_html=True)
        st.write(f"""<p class="info-text">Bottom Pressure: +{impacts['bottom_pressure']:.1f}%</p>""", unsafe_allow_html=True)
        st.write(f"""<p class="info-text">Feed pH: +{impacts['feed_ph']:.1f}%</p>""", unsafe_allow_html=True)

# Operating Guidelines
st.markdown("""<p class="subheading">Operating Guidelines</p>""", unsafe_allow_html=True)
st.markdown("""<ul class="info-text">
<li class="info-text"> Maintain T2-T4 within 195-205°C for optimal conversion</li>
<li class="info-text"> Keep pressure differential balanced for system stability</li>
<li class="info-text"> Monitor pH within optimal range of 6-8</li>
<li class="info-text"> Maintain proper H2:GLY ratio for reaction efficiency</li>
</ul>""", unsafe_allow_html=True)

# Add some spacing
''

st.subheader('Glycerin to Glycol Yield', divider='red')

# fit simple linear regression model
linear_model = ols('GlycerolConversionwt ~ H2GLYMolarRatio+ LiquidFeedSetpointmLmin + PumpLiquidFeedmLmin + MeasuredLiquidFeedmLmin + LHSV1husescatalystvolume + HydrogenGasFlowLmin + TopReactorPressurepsi + BottomReactorPressurepsi + T1C + T2C + T3C + T4C + T5C + HighTemperatureC + AverageTemperatureofT2T3T4C + FeedpH + ProductpH + GlycerolinFeedgkg', data=lrcg1_df).fit()

# display model summary
print(linear_model.summary())

# modify figure size
fig = plt.figure(figsize=(14, 14))

# creating regression plots
fig = sm.graphics.plot_regress_exog(linear_model, 'T2C', fig=fig)

st.write(fig)


#comment this out for now (example of GDP graph and slider)
_ ="""

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

        """