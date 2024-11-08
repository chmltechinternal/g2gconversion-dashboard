import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
from scipy.special import expit

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
    max_value = 200
    range_T2 = st.slider(
    'T2 (°C) - Critical:',
    min_value=0,
    max_value=200,
    value=[min_value, max_value])
    
    range_T3 = st.slider(
    'T3 (°C):',
    min_value=0,
    max_value=200,
    value=[min_value, max_value])
    
    range_T4 = st.slider(
    'T4 (°C):',
    min_value=0,
    max_value=201,
    value=[min_value, max_value])
    
    range_T1 = st.slider(
    'T1 (°C):',
    min_value=0,
    max_value=195,
    value=[min_value, max_value])
    
    range_T5 = st.slider(
    'T5 (°C):',
    min_value=0,
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
