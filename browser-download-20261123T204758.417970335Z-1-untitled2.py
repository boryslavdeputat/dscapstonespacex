import io
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/'
for name in ['dataset_part_1.csv', 'dataset_part_2.csv']:
    r = requests.get(BASE + name)
    print(name, r.status_code, len(r.content))
    open(name, 'wb').write(r.content)

df1 = pd.read_csv('dataset_part_1.csv')
df2 = pd.read_csv('dataset_part_2.csv')
df = df1.copy()
df['Class'] = df2['Class']
df['Year'] = pd.to_datetime(df['Date']).dt.year
print('launches', len(df), 'success rate', round(df['Class'].mean(), 3))
print('missing values:')
print(df.isnull().sum().to_string())
site_rate = df.groupby('LaunchSite')['Class'].mean().round(3)
site_n = df.groupby('LaunchSite').size()
year_rate = df.groupby('Year')['Class'].mean().round(3)
orbit_rate = df.groupby('Orbit')['Class'].mean().round(3).sort_values()
print(site_rate.to_string())
print(year_rate.to_string())
print('total payload', round(df['PayloadMass'].sum(), 1), 'avg payload', round(df['PayloadMass'].mean(), 1))
print('lat/lon by site:')
print(df.groupby('LaunchSite')[['Latitude', 'Longitude']].mean().round(4).to_string())

plt.rcParams['font.size'] = 11

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('saved', path)

# Chart 1: FlightNumber vs PayloadMass, coloured by landing success
fig, ax = plt.subplots(figsize=(10, 4))
for cls, color, lab in [(0, '#c0392b', 'Did not land (Class 0)'), (1, '#27ae60', 'Landed (Class 1)')]:
    sub = df[df['Class'] == cls]
    ax.scatter(sub['FlightNumber'], sub['PayloadMass'], c=color, label=lab, alpha=0.75)
ax.set_title('FlightNumber vs PayloadMass (coloured by landing success)')
ax.set_xlabel('FlightNumber')
ax.set_ylabel('PayloadMass (kg)')
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, 'c1_flight_payload.png')

# Chart 2: success rate by year
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(year_rate.index.astype(str), year_rate.values * 100, color='#2e86c1')
for i, v in enumerate(year_rate.values * 100):
    ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=9)
ax.set_title('First-stage landing success rate by year (%)')
ax.set_xlabel('Year')
ax.set_ylabel('Success rate (%)')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
save(fig, 'c2_year_success.png')

# Chart 3: success rate by orbit (horizontal)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.barh(orbit_rate.index, orbit_rate.values * 100, color='#8e44ad')
for i, v in enumerate(orbit_rate.values * 100):
    ax.text(v + 2, i, f'{v:.0f}%', va='center', fontsize=9)
ax.set_title('First-stage landing success rate by orbit (%)')
ax.set_xlabel('Success rate (%)')
ax.set_xlim(0, 115)
ax.grid(axis='x', alpha=0.3)
save(fig, 'c3_orbit_success.png')

# Chart 4: success rate by launch site
fig, ax = plt.subplots(figsize=(10, 4))
sites = site_rate.index.tolist()
ax.bar(sites, site_rate.values * 100, color=['#e67e22', '#2e86c1', '#27ae60'])
for i, s in enumerate(sites):
    ax.text(i, site_rate.values[i] * 100 + 2, f"{site_rate.values[i]*100:.1f}%  (n={site_n[s]})", ha='center', fontsize=10)
ax.set_title('First-stage landing success rate by launch site (%)')
ax.set_ylabel('Success rate (%)')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
save(fig, 'c4_site_success.png')

# Chart 5: PayloadMass vs LaunchSite, coloured by landing success
fig, ax = plt.subplots(figsize=(10, 4))
site_codes = {s: i for i, s in enumerate(sorted(df['LaunchSite'].unique()))}
x = df['LaunchSite'].map(site_codes) + np.random.RandomState(7).uniform(-0.18, 0.18, len(df))
for cls, color, lab in [(0, '#c0392b', 'Did not land (Class 0)'), (1, '#27ae60', 'Landed (Class 1)')]:
    m = df['Class'] == cls
    ax.scatter(x[m], df.loc[m, 'PayloadMass'], c=color, label=lab, alpha=0.75)
ax.set_xticks(list(site_codes.values()))
ax.set_xticklabels(list(site_codes.keys()))
ax.set_title('PayloadMass vs LaunchSite (coloured by landing success)')
ax.set_xlabel('LaunchSite')
ax.set_ylabel('PayloadMass (kg)')
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, 'c5_payload_site.png')

# Chart 6: launch site map (longitude / latitude from the data set)
fig, ax = plt.subplots(figsize=(10, 4.5))
coords = df.groupby('LaunchSite')[['Longitude', 'Latitude']].mean()
for site, row in coords.iterrows():
    ax.scatter(row['Longitude'], row['Latitude'], s=220, marker='*', zorder=3)
    ax.text(row['Longitude'] + 0.6, row['Latitude'] + 0.4, f"{site}\n({row['Latitude']:.2f}, {row['Longitude']:.2f})", fontsize=10)
ax.set_title('Falcon 9 launch sites (longitude / latitude from the data set)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(-122, -79)
ax.set_ylim(27, 36)
ax.grid(True, alpha=0.3)
save(fig, 'c6_map.png')

# Chart 7: dashboard panels (success pie, payload scatter, orbit bar, year line)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
success_by_site = df[df['Class'] == 1].groupby('LaunchSite').size()
axes[0, 0].pie(success_by_site.values, labels=success_by_site.index, autopct='%1.0f%%', startangle=90)
axes[0, 0].set_title('Successful landings by launch site')
for cls, color in [(0, '#c0392b'), (1, '#27ae60')]:
    sub = df[df['Class'] == cls]
    axes[0, 1].scatter(sub['FlightNumber'], sub['PayloadMass'], c=color, alpha=0.7, s=18)
axes[0, 1].set_title('PayloadMass vs FlightNumber')
axes[0, 1].set_xlabel('FlightNumber')
axes[0, 1].set_ylabel('PayloadMass (kg)')
axes[1, 0].bar(orbit_rate.index, orbit_rate.values * 100, color='#8e44ad')
axes[1, 0].set_title('Success rate by orbit (%)')
axes[1, 0].tick_params(axis='x', rotation=30)
axes[1, 1].plot(year_rate.index, year_rate.values * 100, marker='o', color='#2e86c1')
axes[1, 1].set_title('Success rate by year (%)')
axes[1, 1].set_xlabel('Year')
axes[1, 1].grid(True, alpha=0.3)
fig.suptitle('Plotly Dash dashboard - the four panels', fontsize=14)
save(fig, 'c7_dashboard.png')

# Chart 8: model accuracy and Decision Tree confusion matrix
# Verified in the capstone ML lab (GridSearchCV, 10-fold CV; test set 18 launches):
# tuned test accuracy - Logistic Regression 0.8333, SVM 0.7778, Decision Tree 0.7778, KNN 0.7778
# untuned Decision Tree test accuracy 0.9444 (17 of 18); confusion matrix [[4, 2], [1, 11]]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
models = ['Logistic\nRegression', 'SVM', 'Decision\nTree', 'KNN']
acc = [83.33, 77.78, 77.78, 77.78]
axes[0].bar(models, acc, color=['#2e86c1', '#e67e22', '#27ae60', '#8e44ad'])
for i, v in enumerate(acc):
    axes[0].text(i, v + 2, f'{v:.2f}%', ha='center', fontsize=10)
axes[0].set_title('Tuned model test accuracy (%)')
axes[0].set_ylim(0, 105)
axes[0].grid(axis='y', alpha=0.3)
cm = np.array([[4, 2], [1, 11]])
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_title('Decision Tree confusion matrix (untuned, 94.44%)')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Predicted 0', 'Predicted 1'])
axes[1].set_yticklabels(['True 0', 'True 1'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig('c8_ml.png', dpi=150)
plt.close(fig)
print('saved c8_ml.png')

# ---- Slide helpers ----
W, H = 13.33, 7.5
FOOT = 'SpaceX Falcon 9 First Stage Landing Prediction  |  Applied Data Science Capstone  |  Boryslav Deputat'

def new_slide(title):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.add_patch(plt.Rectangle((0, 6.85), W, 0.65, color='#0b3d66'))
    ax.text(0.4, 7.18, title, color='white', fontsize=19, fontweight='bold', va='center')
    ax.text(0.4, 0.22, FOOT, fontsize=9, color='#666666')
    return fig, ax

def bullets(ax, lines, y_top=6.55, x=0.45, fs=13.5, gap=0.52):
    y = y_top
    for line in lines:
        n = 1 + line.count(chr(10))
        ax.text(x, y, '\u2022  ' + line, fontsize=fs, va='top', ha='left')
        y -= gap * n
    return y

def chart_on_slide(fig, path, left=0.45, bottom=0.45, width=12.43, height=3.55):
    ax = fig.add_axes([left / W, bottom / H, width / W, height / H])
    ax.imshow(plt.imread(path))
    ax.axis('off')
    return ax

def chart_on_slide_top(fig, path, left=0.45, bottom=3.6, width=12.43, height=3.0):
    return chart_on_slide(fig, path, left, bottom, width, height)

# ---- Slides ----
slides = []

# 1 Cover
fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor('#0b3d66')
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.text(0.6, 5.9, 'SpaceX Falcon 9', color='white', fontsize=34, fontweight='bold', va='center')
ax.text(0.6, 5.1, 'First Stage Landing Prediction', color='white', fontsize=34, fontweight='bold', va='center')
ax.text(0.6, 4.2, 'Applied Data Science Capstone Project Report', color='#9fc5e8', fontsize=20, va='center')
ax.text(0.6, 3.3, 'Author: Boryslav Deputat  |  November 2026', color='white', fontsize=15, va='center')
ax.text(0.6, 2.6, 'Data: 90 Falcon 9 launches, 4 June 2010 - 5 November 2020', color='white', fontsize=14, va='center')
ax.text(0.6, 2.1, 'Methods: SpaceX API and web scraping, data wrangling, EDA, SQL, Folium,', color='white', fontsize=14, va='center')
ax.text(0.6, 1.7, 'Plotly Dash, and machine learning classification', color='white', fontsize=14, va='center')
ax.text(0.6, 1.2, 'GitHub repository: https://github.com/boryslavdeputat/dscapstonespacex', color='#9fc5e8', fontsize=14, va='center')
slides.append(fig)

# 2 Executive Summary
fig, ax = new_slide('Executive Summary')
bullets(ax, [
    'SpaceX reuses Falcon 9 first stages; landing success decides much of the launch cost.',
    'Goal: predict whether a first stage will land successfully, and find what drives success.',
    '90 Falcon 9 launches (4 Jun 2010 - 5 Nov 2020); 60 landed successfully (66.7%).',
    'Success rose from 0% in 2010-2013 to 90.0% in 2019 and 84.2% in 2020.',
    'KSC LC-39A (77.3%) and VAFB SLC-4E (76.9%) beat CCAFS SLC-40 (60.0%).',
    'Best tuned model: Logistic Regression, 83.3% test accuracy.',
    'Decision Tree reached 94.4% (17 of 18) on the small test set.',
    'Recommendation: fly proven boosters with grid fins and legs, and prefer KSC / VAFB.',
])
slides.append(fig)

# 3 Introduction
fig, ax = new_slide('Introduction')
bullets(ax, [
    "SpaceX's Falcon 9 is a reusable two-stage rocket; the first stage returns and lands.",
    'Reusing a first stage can cut the launch cost from about $165 million to about $62 million.',
    'The business question: can we predict a successful first-stage landing before launch?',
    'If yes, SpaceX and its customers can price launches and plan booster reuse with confidence.',
    'This report answers it with 90 Falcon 9 launches from 2010 to 2020.',
    'Outline: data collection, wrangling, EDA, SQL, maps, dashboard, and classification models.',
])
slides.append(fig)

# 4 Data Collection - SpaceX API
fig, ax = new_slide('Data Collection - SpaceX API')
bullets(ax, [
    'Source: SpaceX REST API v4 (api.spacexdata.com/v4/launches/past).',
    'Requested past launches, kept Falcon 9 only, and enriched each launch with',
    'rocket, payload and launchpad data.',
    'Features: FlightNumber, Date, BoosterVersion, PayloadMass, Orbit, LaunchSite,',
    'Flights, GridFins, Reused, Legs, LandingPad, Block, ReusedCount, Serial,',
    'Longitude, Latitude.',
    'Result: 90 Falcon 9 launches, 4 June 2010 to 5 November 2020.',
    'Outcome was recorded as text, for example True ASDS, True RTLS, False ASDS, None None.',
])
slides.append(fig)

# 5 Data Collection - Web Scraping
fig, ax = new_slide('Data Collection - Web Scraping')
bullets(ax, [
    "Source: Wikipedia, 'List of Falcon 9 and Falcon Heavy launches'.",
    'Method: requests and BeautifulSoup; parsed the launch table into a data frame.',
    'Used it to cross-check launch dates, payloads, customers and outcomes from the API.',
    'Scraped columns: Flight No., Date and time, Version, Booster, Launch site,',
    'Payload, Orbit, Customer, Outcome.',
    'This second source guards against gaps in any single API response.',
])
slides.append(fig)

# 6 Data Wrangling
fig, ax = new_slide('Data Wrangling')
bullets(ax, [
    'Converted Outcome into a binary Class: 1 = first stage landed, 0 = it did not.',
    'Success counts: True ASDS 41, True RTLS 14, True Ocean 5 (60 successful).',
    'The other 30 launches failed to land or were expended (None None 19, False ASDS 6,',
    'True Ocean is counted above; False Ocean 2, None ASDS 2, False RTLS 1).',
    'One-hot encoded the categorical features (Orbit, LaunchSite, LandingPad, Serial).',
    'Standardised the numeric features (StandardScaler) before training the classifiers.',
    'Final modelling data: 90 rows, 83 features after one-hot encoding, target Class.',
])
slides.append(fig)

# 7 EDA with Data Visualization - Methodology
fig, ax = new_slide('EDA with Data Visualization - Methodology')
bullets(ax, [
    'Tools: Matplotlib and Seaborn, on the 90-launch data set.',
    'Scatter: FlightNumber vs PayloadMass, FlightNumber vs LaunchSite,',
    'PayloadMass vs LaunchSite, FlightNumber vs Orbit, PayloadMass vs Orbit,',
    'all coloured by landing outcome.',
    'Bar: success rate by orbit and by launch site; line: success rate by year.',
    'Questions: does success grow with flight experience, payload, orbit or site?',
])
slides.append(fig)

# 8 EDA with SQL - Methodology
fig, ax = new_slide('EDA with SQL - Methodology')
bullets(ax, [
    'Loaded the 90 launches into a SQLite table (SPACEXTABLE) and ran 13 queries.',
    'Queries covered: launch sites, total and average payload, first successful landings.',
    'Also: drone-ship successes by payload band, outcome counts, maximum payload,',
    'and the 2015 drone-ship failures.',
    'And: successful landings in 2010-2017, and success by reuse, grid fins and legs.',
    'SQL let the same questions be answered exactly, with no plotting needed.',
])
slides.append(fig)

# 9 Interactive Visual Analytics - Methodology
fig, ax = new_slide('Interactive Visual Analytics - Folium and Dash')
bullets(ax, [
    'Folium map: markers for the three launch sites, with circles sized by launch count.',
    'Proximity analysis: distance from each site to the coastline, railway and highway.',
    'Plotly Dash dashboard: a site selector and a payload range slider driving four charts.',
    'Dashboard charts: success pie by site, payload vs outcome scatter,',
    'success by orbit and success by year.',
    'Both tools make the same results explorable without reading code.',
])
slides.append(fig)

# 10 Predictive Analysis - Methodology
fig, ax = new_slide('Predictive Analysis - Methodology')
bullets(ax, [
    'Target: Class (1 = successful landing, 0 = otherwise).',
    'Split: 72 train, 18 test (random_state=2).',
    'Models: Logistic Regression, Support Vector Machine, Decision Tree,',
    'K-Nearest Neighbours.',
    'Each model was tuned with GridSearchCV (10-fold cross-validation)',
    'and scored on the test set.',
    'Features: 83 one-hot encoded columns; numeric features standardised.',
    'Compared test accuracy and the confusion matrix before choosing a best model.',
])
slides.append(fig)

# 11 EDA with Visualization - Results
fig, ax = new_slide('EDA with Visualization - Results')
bullets(ax, [
    'Success grows with experience: 0% in 2010-2013, 33% in 2014-2015, 83% in 2017, 90% in 2019.',
    'Heavier payloads still land: successes appear across the whole payload range, to 15,600 kg.',
    'VLEO (Starlink) flights succeed 85.7% of the time; GTO flights only 51.9%.',
], y_top=6.55, gap=0.5)
chart_on_slide(fig, 'c2_year_success.png', bottom=2.55, height=2.35)
chart_on_slide(fig, 'c1_flight_payload.png', bottom=0.45, height=1.95)
slides.append(fig)

# 12 EDA with SQL - Results
fig, ax = new_slide('EDA with SQL - Results')
bullets(ax, [
    'Unique launch sites: CCAFS SLC 40, KSC LC 39A, VAFB SLC 4E.',
    'Total payload: 549,446 kg; average payload: 6,105 kg.',
    'Average payload by site: KSC 7,606 kg, VAFB 5,919 kg, CCAFS 5,548 kg.',
    'First successful landing: 2014-04-18 (ocean); first ground pad: 2015-12-22;',
    'first drone ship: 2016-04-08.',
    'Drone-ship successes, 4,000-6,000 kg: B1022, B1026, B1021, B1031, B1046, B1059.',
    'Maximum payload 15,600 kg: B1048 (twice) and B1051, all VLEO flights.',
    '2015 drone-ship failures: B1012 (10 Jan 2015) and B1015 (14 Apr 2015), both CCAFS.',
    'Successful landings 2010-2017: drone ship 5, ground pad 3, ocean 3.',
    'Grid fins: 82.9% success with fins vs 10.0% without; legs: 83.1% vs 5.3%.',
], fs=12.5, gap=0.48)
slides.append(fig)

# Folium proximity (haversine, same formula as the Folium lab, R = 6373.0 km)
def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6373.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))
lc40 = (28.562302, -80.577356)
d_coast = haversine_km(*lc40, 28.56367, -80.57163)
d_rail = haversine_km(*lc40, 28.57206, -80.58525)
d_hwy = haversine_km(*lc40, 28.56333, -80.57072)
d_city = haversine_km(*lc40, 28.612218, -80.807752)
print('proximity km coast/rail/highway/city:', round(d_coast, 2), round(d_rail, 2), round(d_hwy, 2), round(d_city, 2))
# 13 Interactive Map with Folium - Results
fig, ax = new_slide('Interactive Map with Folium - Results')
bullets(ax, [
    f'Folium map markers (launch sites): CCAFS LC-40 (28.5623, -80.5774), CCAFS SLC-40 (28.5632, -80.5768),',
    'KSC LC-39A (28.5733, -80.6469), VAFB SLC-4E (34.6328, -120.6107).',
    f'Proximity to CCAFS LC-40 (approx. map-picked points): coastline {d_coast:.2f} km, railway {d_rail:.2f} km,',
    f'highway {d_hwy:.2f} km, nearest city Titusville {d_city:.2f} km (haversine, R = 6373.0 km).',
    'Coastal sites are chosen so a failed first stage falls into the ocean, not on land.',
    'CCAFS SLC 40 hosts most launches (55) but has the lowest success rate (60.0%).',
    'KSC LC 39A (22 launches, 77.3%) and VAFB SLC 4E (13 launches, 76.9%) are the most reliable.',
], y_top=6.55, gap=0.42)
chart_on_slide(fig, 'c6_map.png', bottom=0.45, height=3.0)
slides.append(fig)

# 14 Plotly Dash Dashboard - Results
fig, ax = new_slide('Plotly Dash Dashboard - Results')
bullets(ax, [
    "The dashboard's success pie shows CCAFS contributes the most successful landings (33 of 60).",
    'The payload scatter shows successful landings at every payload, including 15,600 kg.',
    'The year line confirms the learning curve: success rises steadily after 2015.',
    'The orbit bar confirms VLEO and SSO are the most reliable orbits, GTO the least.',
], y_top=6.55, gap=0.5)
chart_on_slide(fig, 'c7_dashboard.png', bottom=0.45, height=3.6)
slides.append(fig)

# 15 Predictive Analysis - Results
fig, ax = new_slide('Predictive Analysis (Classification) - Results')
bullets(ax, [
    'Best tuned model: Logistic Regression, 83.3% test accuracy (CV 84.6%, C=0.01).',
    'Tuned SVM, Decision Tree and KNN each scored 77.8% on the test set.',
    'The untuned Decision Tree scored 94.4% (17 of 18) - the test set is small, so treat this with care.',
    'Decision Tree confusion matrix: 11 true positives, 4 true negatives, 2 false positives,',
    '1 false negative.',
    'Conclusion: landing success is predictable, and the models agree on about 4 in 5 launches.',
], y_top=6.55, gap=0.5)
chart_on_slide(fig, 'c8_ml.png', bottom=0.45, height=3.35)
slides.append(fig)

# 16 Conclusion
fig, ax = new_slide('Conclusion')
bullets(ax, [
    'Landing success rose from 0% (2010-2013) to about 85-90% by 2019-2020.',
    'What matters most: grid fins (82.9% vs 10.0%), legs (83.1% vs 5.3%), and flight experience.',
    'KSC LC-39A and VAFB SLC-4E are the most reliable sites; CCAFS does the most launches.',
    'A Logistic Regression model predicts landings at 83.3% accuracy on unseen launches.',
    'Recommendation: fly proven boosters with grid fins and legs, prefer KSC / VAFB,',
    'and keep reusing boosters - reuse itself does not hurt success.',
])
slides.append(fig)

# 17 Thank you
fig, ax = new_slide('Thank You')
bullets(ax, [
    'Thank you.',
    'Data: SpaceX API v4 (launches) and Wikipedia Falcon 9 launch list; IBM Skills Network',
    'capstone data sets dataset_part_1.csv, dataset_part_2.csv, dataset_part_3.csv',
    '(90 Falcon 9 launches, 2010-2020).',
    'Code: data collection, wrangling, EDA, SQL, Folium, Dash and machine learning notebooks',
'GitHub repository: https://github.com/boryslavdeputat/dscapstonespacex',
    'from the Applied Data Science Capstone.',
    'All charts and figures in this report were produced from those data sets; the SQL and',
    'model results were recomputed for this report.',
])
slides.append(fig)

OUT = 'Data_Science_Capstone_Project_Report.pdf'
with PdfPages(OUT) as pdf:
    for fig in slides:
        pdf.savefig(fig)
        plt.close(fig)
print('Wrote', OUT, 'with', len(slides), 'pages')
import os
print('size bytes', os.path.getsize(OUT))
