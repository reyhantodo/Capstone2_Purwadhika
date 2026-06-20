# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('VicRoad_Crash_Data_Final.csv')
df

# %%
df.shape

# %%
df.info()

# %%
df.describe().T

# %%
#starting check data cleaned or not

# %%
df.isna().sum()

# %%
df.duplicated().sum()

# %%
print("Duplicate ACCIDENT_NO:")
print(df['ACCIDENT_NO'].duplicated().sum())

# %%
day_consistency = df.groupby('DAY_OF_WEEK')['DAY_WEEK_DESC'].nunique()

print(day_consistency)
#day_of_week udah konsisten
#EN1

# %%
pd.crosstab(df['DAY_OF_WEEK'], df['DAY_WEEK_DESC'])

# %%
day_table = pd.crosstab(df['DAY_OF_WEEK'], df['DAY_WEEK_DESC'])

print("Crosstab total:", day_table.sum().sum())
print("Dataframe total:", len(df))

# %%
#final cross check
# 1. Missing values
print("Missing values:")
print(df.isna().sum())

# 2. Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# 3. Duplicate accident numbers
print("\nDuplicate ACCIDENT_NO:")
print(df['ACCIDENT_NO'].duplicated().sum())

# 4. Date conversion check
date_check = pd.to_datetime(
    df['ACCIDENT_DATE'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

print("\nInvalid dates:")
print(date_check.isna().sum())

# 5. Day consistency check
day_table = pd.crosstab(df['DAY_OF_WEEK'], df['DAY_WEEK_DESC'])

print("\nDay crosstab total:", day_table.sum().sum())
print("Dataframe total:", len(df))

# 6. Accident type code-description consistency
print("\nACCIDENT_TYPE consistency issue:")
accident_type_check = df.groupby('ACCIDENT_TYPE')['ACCIDENT_TYPE_DESC'].nunique()
print(accident_type_check[accident_type_check > 1])

# 7. DCA code-description consistency
print("\nDCA_CODE consistency issue:")
dca_check = df.groupby('DCA_CODE')['DCA_DESC'].nunique()
print(dca_check[dca_check > 1])

# %%
for col in ['SEVERITY', 'SPEED_ZONE', 'LIGHT_CONDITION', 'ROAD_GEOMETRY_DESC', 'RMA']:
    print(f"\n--- {col} ---")
    print(df[col].value_counts().sort_index())

# %%
unknown_count = (df['ROAD_GEOMETRY_DESC'] == 'Unknown').sum()
total_count = len(df)

unknown_percentage = (unknown_count / total_count) * 100

print("Unknown count:", unknown_count)
print("Total rows:", total_count)
print(f"Unknown percentage: {unknown_percentage:.2f}%")
#persentase tidak signifikan

# %%
from scipy.stats import shapiro

numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    sample = df[col].dropna().sample(
        min(5000, len(df[col].dropna())), 
        random_state=42
    )
    
    stat, p_value = shapiro(sample)
    
    print(f"\nColumn: {col}")
    print(f"p-value: {p_value}")
    
    if p_value > 0.05:
        print("Result: Normally distributed")
    else:
        print("Result: Not normally distributed")

# %%
numerical_features = ['ACCIDENT_TYPE', 'DCA_CODE', 'NODE_ID', 'DAY_OF_WEEK', 'LIGHT_CONDITION', 'SEVERITY', 'SPEED_ZONE']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# %%
from scipy.stats import shapiro
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Select column
speed = df['SPEED_ZONE']

# 1. Countplot
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df,
    x='SPEED_ZONE',
    order=sorted(df['SPEED_ZONE'].unique())
)
plt.title('Distribution of Speed Zone')
plt.xlabel('Speed Zone')
plt.ylabel('Number of Accidents')
plt.show()

# 2. Q-Q plot
stats.probplot(speed, dist="norm", plot=plt)
plt.title('Q-Q Plot of Speed Zone')
plt.show()

# 3. Shapiro-Wilk test using sample
speed_sample = speed.sample(5000, random_state=42)

stat, p_value = shapiro(speed_sample)

print("Shapiro Statistic:", stat)
print("p-value:", p_value)

if p_value > 0.05:
    print("Conclusion: Speed Zone appears normally distributed")
else:
    print("Conclusion: Speed Zone is not normally distributed")
#EN2

# %%
#start data insight
import pandas as pd
import plotly.express as px

#change type data temporary supaya bisa digunakan
df['ACCIDENT_DATE'] = pd.to_datetime(
    df['ACCIDENT_DATE'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

df['YEAR'] = df['ACCIDENT_DATE'].dt.year
df['MONTH'] = df['ACCIDENT_DATE'].dt.month
df['MONTH_NAME'] = df['ACCIDENT_DATE'].dt.month_name()

time_check = pd.to_datetime(
    df['ACCIDENT_TIME'].astype(str),
    errors='coerce'
)

df['HOUR'] = time_check.dt.hour

# %%
df.info()

# %%
#insight1
import plotly.express as px

counts = df['DCA_DESC'].value_counts().reset_index()
counts.columns = ['DCA_DESC', 'Count']

# Sort descending
counts = counts.sort_values('Count', ascending=False)

fig = px.bar(
    counts,
    x='Count',
    y='DCA_DESC',
    orientation='h',
    height=3000,
    text='Count'
)

fig.update_traces(textposition='outside')

fig.update_layout(
    title='Distribution of Accident Types',
    xaxis_title='Number of Accidents',
    yaxis_title='Accident Description',
    yaxis={'categoryorder':'total ascending'}
)
fig.show()

# %%
#insight 2
severity_counts = df['SEVERITY'].value_counts().sort_index().reset_index()
severity_counts.columns = ['SEVERITY', 'Accident_Count']

severity_counts['Percentage'] = (
    severity_counts['Accident_Count'] / severity_counts['Accident_Count'].sum() * 100
)

fig = px.bar(
    severity_counts,
    x='SEVERITY',
    y='Accident_Count',
    text=severity_counts['Percentage'].round(2).astype(str) + '%',
    title='Accident Severity Distribution'
)

fig.update_layout(
    xaxis_title='Severity Code',
    yaxis_title='Number of Accidents'
)

fig.show()

# %%
#insight 3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

speed_severity_table = pd.crosstab(df['SPEED_ZONE'], df['SEVERITY'])

plt.figure(figsize=(12, 6))
sns.heatmap(
    speed_severity_table,
    annot=True,
    fmt='g',
    cmap='Reds',
    linewidths=0.5,
    cbar_kws={'label': 'Accident Count'}
)

plt.title('Accident Count by Speed Zone and Severity', fontsize=14)
plt.xlabel('Severity', fontsize=12)
plt.ylabel('Speed Zone', fontsize=12)
plt.yticks(rotation=0)
plt.show()

# %%
#insight 4
hour_counts = df['HOUR'].value_counts().reindex(range(24), fill_value=0).reset_index()
hour_counts.columns = ['HOUR', 'Accident_Count']

fig = px.line(
    hour_counts,
    x='HOUR',
    y='Accident_Count',
    markers=True,
    title='Number of Accidents by Hour of Day'
)

fig.update_layout(
    xaxis_title='Hour of Day',
    yaxis_title='Number of Accidents'
)

fig.show()

# %%
#tambahan
def time_period(hour):
    if 5 <= hour <= 9:
        return 'Morning Peak'
    elif 10 <= hour <= 14:
        return 'Midday'
    elif 15 <= hour <= 18:
        return 'Afternoon Peak'
    elif 19 <= hour <= 23:
        return 'Night'
    else:
        return 'Late Night'

df['TIME_PERIOD'] = df['HOUR'].apply(time_period)

time_order = [
    'Late Night',
    'Morning Peak',
    'Midday',
    'Afternoon Peak',
    'Night'
]

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x='TIME_PERIOD',
    order=time_order
)

plt.title('Number of Accidents by Time Period')
plt.xlabel('Time Period')
plt.ylabel('Number of Accidents')
plt.show()

# %%
#insight 5
day_order = [
    'Monday', 'Tuesday', 'Wednesday',
    'Thursday', 'Friday', 'Saturday', 'Sunday'
]

day_counts = df['DAY_WEEK_DESC'].value_counts().reindex(day_order).reset_index()
day_counts.columns = ['DAY_WEEK_DESC', 'Accident_Count']

fig = px.bar(
    day_counts,
    x='DAY_WEEK_DESC',
    y='Accident_Count',
    text='Accident_Count',
    title='Number of Accidents by Day of Week'
)

fig.update_layout(
    xaxis_title='Day of Week',
    yaxis_title='Number of Accidents'
)

fig.show()

# %%
#insight 6
df['DAY_TYPE'] = df['DAY_WEEK_DESC'].apply(
    lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday'
)

day_type_counts = df['DAY_TYPE'].value_counts().reset_index()
day_type_counts.columns = ['DAY_TYPE', 'Accident_Count']

fig = px.bar(
    day_type_counts,
    x='DAY_TYPE',
    y='Accident_Count',
    text='Accident_Count',
    title='Weekday vs Weekend Accident Count'
)

fig.update_layout(
    xaxis_title='Day Type',
    yaxis_title='Number of Accidents'
)

fig.show()

# %%
#insight 7
monthly = df.groupby(df['ACCIDENT_DATE'].dt.to_period('M')).size().reset_index()
monthly.columns = ['YEAR_MONTH', 'Accident_Count']
monthly['YEAR_MONTH'] = monthly['YEAR_MONTH'].astype(str)

fig = px.line(
    monthly,
    x='YEAR_MONTH',
    y='Accident_Count',
    markers=True,
    title='Monthly Accident Trend from 2019 to 2023'
)

fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Number of Accidents'
)

fig.show()

# %%
#insight 8
yearly = df['YEAR'].value_counts().sort_index().reset_index()
yearly.columns = ['YEAR', 'Accident_Count']

fig = px.line(
    yearly,
    x='YEAR',
    y='Accident_Count',
    markers=True,
    title='Yearly Accident Trend'
)

fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Number of Accidents'
)

fig.show()

# %%
#insight 9
# Count table
light_count = pd.crosstab(
    df['LIGHT_CONDITION'],
    df['SEVERITY']
).sort_index()

# Percentage table
light_pct = pd.crosstab(
    df['LIGHT_CONDITION'],
    df['SEVERITY'],
    normalize='index'
).sort_index() * 100

# Create annotation labels: count + percentage
labels = light_count.astype(str) + "\n(" + light_pct.round(1).astype(str) + "%)"

plt.figure(figsize=(11, 6))

sns.heatmap(
    light_count,
    annot=labels,
    fmt='',
    cmap='Reds',
    linewidths=0.5,
    cbar_kws={'label': 'Accident Count'}
)

plt.title('Accident Count and Severity Percentage by Light Condition', fontsize=14)
plt.xlabel('Severity', fontsize=12)
plt.ylabel('Light Condition', fontsize=12)
plt.yticks(rotation=0)
plt.show()
#EN3

# %%
#insight 10
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Count table
road_count = pd.crosstab(
    df['ROAD_GEOMETRY_DESC'],
    df['SEVERITY']
)

# Percentage table by road geometry
road_pct = pd.crosstab(
    df['ROAD_GEOMETRY_DESC'],
    df['SEVERITY'],
    normalize='index'
) * 100

# Create labels: count + percentage
labels = road_count.astype(str) + "\n(" + road_pct.round(1).astype(str) + "%)"

plt.figure(figsize=(13, 7))

sns.heatmap(
    road_count,
    annot=labels,
    fmt='',
    cmap='Reds',
    linewidths=0.5,
    cbar_kws={'label': 'Accident Count'}
)

plt.title('Accident Count and Severity Percentage by Road Geometry', fontsize=14)
plt.xlabel('Severity', fontsize=12)
plt.ylabel('Road Geometry', fontsize=12)
plt.yticks(rotation=0)
plt.show()

# %%
#insight 11
#conv lat long to num
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

#Remove missing coordinates
df_map = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

#most common value
def most_common(x):
    return x.mode().iloc[0] if not x.mode().empty else None

#group by NODE_ID
node_hotspots = df_map.groupby('NODE_ID').agg(
    Accident_Count=('ACCIDENT_NO', 'count'),
    LATITUDE=('LATITUDE', 'mean'),
    LONGITUDE=('LONGITUDE', 'mean'),
    ROAD_NAME=('ROAD_NAME', most_common),
    LGA_NAME=('LGA_NAME', most_common),
    Most_Common_Accident_Type=('DCA_DESC', most_common),
    Most_Common_Severity=('SEVERITY', most_common)
).reset_index()

#greate accident count category
node_hotspots['Hotspot_Level'] = pd.cut(
    node_hotspots['Accident_Count'],
    bins=[0, 5, 10, 15, 20, float('inf')],
    labels=[
        'Very Low (1-5)',
        'Low (6-10)',
        'Medium (11-15)',
        'High (16-20)',
        'Very High (20+)'
    ]
)

#convert NODE_ID to text
node_hotspots['NODE_ID'] = node_hotspots['NODE_ID'].astype(str)

fig = px.scatter_mapbox(
    node_hotspots,
    lat='LATITUDE',
    lon='LONGITUDE',
    size='Accident_Count',
    color='Hotspot_Level',
    hover_name='NODE_ID',
    hover_data={
        'ROAD_NAME': True,
        'LGA_NAME': True,
        'Accident_Count': True,
        'Most_Common_Accident_Type': True,
        'Most_Common_Severity': True,
        'LATITUDE': False,
        'LONGITUDE': False
    },
    size_max=30,
    zoom=8,
    height=750,
    title='Accident Hotspot Map by Accident Count Category'
)

fig.update_layout(
    mapbox_style='open-street-map',
    margin={"r":0, "t":50, "l":0, "b":0}
)

fig.show()

# %%
#insight 12
rma_counts = df['RMA'].value_counts().reset_index()
rma_counts.columns = ['RMA', 'Accident_Count']

fig = px.bar(
    rma_counts,
    x='Accident_Count',
    y='RMA',
    orientation='h',
    text='Accident_Count',
    title='Accidents by Road Management Authority'
)

fig.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    xaxis_title='Number of Accidents',
    yaxis_title='RMA'
)

fig.show()

# %%
#get top 10 accident types
top_dca = df['DCA_DESC'].value_counts().head(10).index

#create crosstab: rows = accident type, columns = speed zone
speed_dca_table = pd.crosstab(
    df['DCA_DESC'],
    df['SPEED_ZONE']
)

#keep only top 10 accident types
speed_dca_top = speed_dca_table.loc[top_dca]

#sort speed zone columns
speed_order = sorted(df['SPEED_ZONE'].dropna().unique())
speed_dca_top = speed_dca_top[speed_order]

#wrap long accident descriptions
speed_dca_top.index = [fill(label, width=35) for label in speed_dca_top.index]

#plot heatmap
plt.figure(figsize=(14, 8))

sns.heatmap(
    speed_dca_top,
    annot=True,
    fmt='g',
    cmap='Reds',
    linewidths=0.5,
    cbar_kws={'label': 'Accident Count'}
)

plt.title('Top 10 Accident Types by Speed Zone', fontsize=14)
plt.xlabel('Speed Zone', fontsize=12)
plt.ylabel('Accident Type', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# %%
#revisi start
#time period checking for speed zone factor
speed_time = pd.crosstab(
    df['SPEED_ZONE'],
    df['TIME_PERIOD']
)

plt.figure(figsize=(12, 6))

sns.heatmap(
    speed_time,
    annot=True,
    fmt='g',
    cmap='Reds',
    linewidths=0.5
)

plt.title('Accident Count by Speed Zone and Time Period')
plt.xlabel('Time Period')
plt.ylabel('Speed Zone')
plt.show()

# %%
#top 10 accidents in 60km/h speed zone
speed_dca_60 = df[df['SPEED_ZONE'] == 60]['DCA_DESC'].value_counts().head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=speed_dca_60.values,
    y=speed_dca_60.index
)

plt.title('Top 10 Accident Types in 60 km/h Speed Zone')
plt.xlabel('Accident Count')
plt.ylabel('Accident Type')
plt.show()

# %%
#road geometry check
speed_road_geometry_60 = (
    df[df['SPEED_ZONE'] == 60]['ROAD_GEOMETRY_DESC']
    .value_counts()
)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=speed_road_geometry_60.values,
    y=speed_road_geometry_60.index
)

plt.title('Road Geometry Distribution in 60 km/h Speed Zone')
plt.xlabel('Accident Count')
plt.ylabel('Road Geometry')
plt.show()

# %%
#time period checking for light condition factor
light_time = pd.crosstab(
    df['LIGHT_CONDITION'],
    df['TIME_PERIOD']
)

plt.figure(figsize=(12, 6))

sns.heatmap(
    light_time,
    annot=True,
    fmt='g',
    cmap='Reds',
    linewidths=0.5
)

plt.title('Accident Count by Light Condition and Time Period')
plt.xlabel('Time Period')
plt.ylabel('Light Condition')
plt.show()

# %%
#hour checking for light condition factor
daylight_hour = (
    df[df['LIGHT_CONDITION'] == 1]
    .groupby('HOUR')
    .size()
    .reset_index(name='Accident_Count')
)

daylight_hour

plt.figure(figsize=(10, 5))

sns.lineplot(
    data=daylight_hour,
    x='HOUR',
    y='Accident_Count',
    marker='o'
)

plt.title('Daylight Accident Count by Hour')
plt.xlabel('Hour')
plt.ylabel('Accident Count')
plt.show()


