import branca
import sys
import os
import geopandas as gpd
from scipy import stats
import pandas as pd
import numpy as np
import folium
import math
import requests
import json
import copy
import matplotlib

from os import listdir
from shapely.geometry import Point, LineString

config_input_path = "processed_6_parades_linia_Barcelona_conAlturas.csv"
GEOAPIFY_KEY = os.environ["GEOAPIFY_KEY"]


def create_new_dataset(data, mode, exceptions_index=[], transits_index = []):
    new_data = []
    exceptions_true_index = []
    transits_true_index = []
    get_data = False if mode == 2 else True
    for i, row in data.iterrows():
        if row['Final Stop'] == 1:
            x = float(row["To"].replace("(", "").replace(")", "").split(",")[0])
            y = float(row["To"].replace("(", "").replace(")", "").split(",")[1])
            new_data.append(pd.Series([x, y, 1]))
            if mode == 0:
                get_data = False
            elif mode == 2:
                get_data = True
        if i == 0 and get_data:
            x = float(row["From"].replace("(", "").replace(")", "").split(",")[0])
            y = float(row["From"].replace("(", "").replace(")", "").split(",")[1])
            new_data.append(pd.Series([x, y, 1]))

            x = float(row["To"].replace("(", "").replace(")", "").split(",")[0])
            y = float(row["To"].replace("(", "").replace(")", "").split(",")[1])
            new_data.append(pd.Series([x, y, 1]))
        elif get_data:
            x = float(row["To"].replace("(", "").replace(")", "").split(",")[0])
            y = float(row["To"].replace("(", "").replace(")", "").split(",")[1])
            new_data.append(pd.Series([x, y, 1]))
        if i in exceptions_index and get_data:
            exceptions_true_index.append(len(new_data) - 1)
        if i in transits_index and get_data:
            transits_true_index.append(len(new_data) - 1)
        #if i==34:
        #    break
        
    
       
    new_data = pd.DataFrame(new_data)
    new_data.columns = ["Y", "X", "Bus Stop"]
    return new_data,exceptions_true_index, transits_true_index

def create_new_dataset_full(data, mode, exceptions_index=[], transits_index = [], auxiliary = None):
    new_data = []
    exceptions_true_index = []
    transits_true_index = []
    get_data = False if mode == 2 else True
    cont = 0
    for i, row in auxiliary.iterrows():
        if row['Bus Stop'] == 1:
            if row['Final Stop'] == 1:
                x = float(data.loc[cont,"To"].replace("(", "").replace(")", "").split(",")[0])
                y = float(data.loc[cont,"To"].replace("(", "").replace(")", "").split(",")[1])
                distance = float(row["Distance"])
                altitude = float(row["Altitude To"])
                new_data.append(pd.Series([x, y, 1, distance, altitude]))
                if mode == 0:
                    get_data = False
                elif mode == 2:
                    get_data = True
            if i == 0 and get_data:
                x = float(data.loc[cont, "From"].replace("(", "").replace(")", "").split(",")[0])
                y = float(data.loc[cont, "From"].replace("(", "").replace(")", "").split(",")[1])
                distance = 0
                altitude = float(row["Altitude From"])
                new_data.append(pd.Series([x, y, 1, distance, altitude]))

                x = float(data.loc[cont, "To"].replace("(", "").replace(")", "").split(",")[0])
                y = float(data.loc[cont, "To"].replace("(", "").replace(")", "").split(",")[1])
                altitude = float(row["Altitude To"])
                distance = float(row["Distance"])
                bus_stop = 1
                new_data.append(pd.Series([x, y, 1, distance, altitude]))
            elif i == (len(data) - 1) and get_data:
                x = float(data.loc[cont,"To"].replace("(", "").replace(")", "").split(",")[0])
                y = float(data.loc[cont,"To"].replace("(", "").replace(")", "").split(",")[1])
                altitude = float(row["Altitude To"])
                distance = float(row["Distance"])
                bus_stop = 1
                new_data.append(pd.Series([x, y, 1, distance, altitude]))
            elif get_data:
                x = float(data.loc[cont,"To"].replace("(", "").replace(")", "").split(",")[0])
                y = float(data.loc[cont,"To"].replace("(", "").replace(")", "").split(",")[1])
                altitude = float(row["Altitude To"])
                distance = float(row["Distance"])
                bus_stop = 1
                new_data.append(pd.Series([x, y, 1, distance, altitude]))
            if i in exceptions_index and get_data:
                exceptions_true_index.append(len(new_data) - 1)
            if i in transits_index and get_data:
                transits_true_index.append(len(new_data) - 1)
            cont+=1
        #if i==34:
        #    break
        
    
       
    new_data = pd.DataFrame(new_data)
    new_data.columns = ["Y", "X", "Bus Stop", "distance", "Altitude"]
    
    return new_data,exceptions_true_index, transits_true_index


class MapVisualization:
    def __init__(self, filename, mode, exceptions_index, transits_index, full=False, auxiliary=''):
        ext = filename.split('.')[-1]
        if ext == 'xlsx':
            self.data = pd.read_excel(filename)
        elif ext == 'csv':
            self.data = pd.read_csv(filename, delimiter=None)
        
        if full:
            self.data, self.exceptions_index, self.transits_index = create_new_dataset_full(self.data, mode, exceptions_index, transits_index, pd.read_csv(auxiliary, delimiter=None))
        else:
            self.data, self.exceptions_index, self.transits_index = create_new_dataset(self.data, mode, exceptions_index, transits_index)

    def setGeometry(self):
        X = []
        Y = []

        self.gdf = gpd.GeoDataFrame(self.data, geometry=gpd.points_from_xy(
           self.data["X"], self.data["Y"]), crs=4326)

    def getLines(self, pairs=None):
        lines = []
        self.total_pairs = [item[0] - 1 for item in pairs ]
        self.total_pairs.append(pairs[-1][-1] - 1)
        
        for i in pairs:
            aux = []
            
            # Call the OSMR API
            _from = [self.data["X"][i[0]], self.data["Y"][i[0]]]
            _to = [self.data["X"][i[1]], self.data["Y"][i[1]]]
            
            if i[0] in self.exceptions_index:
                r = requests.get(f"https://api.geoapify.com/v1/routing?waypoints={_from[1]}%2C{_from[0]}%7C{_to[1]}%2C{_to[0]}&mode=bus&apiKey={GEOAPIFY_KEY}""")
                
                try:
                    routes = json.loads(r.content)
                    route_1 = routes.get("features")[0]
                    coordinates = copy.deepcopy(route_1["geometry"]["coordinates"][0])
                except:
                    print(f"https://api.geoapify.com/v1/routing?waypoints={_from[1]}%2C{_from[0]}%7C{_to[1]}%2C{_to[0]}&mode=bus&apiKey={GEOAPIFY_KEY}")
                    coordinates = [_from, _to]
            
            elif i[0] in self.transits_index:
                r = requests.get(f"https://api.geoapify.com/v1/routing?waypoints={_from[1]}%2C{_from[0]}%7C{_to[1]}%2C{_to[0]}&mode=transit&apiKey={GEOAPIFY_KEY}""")
                
                try:
                    routes = json.loads(r.content)
                    route_1 = routes.get("features")[0]
                    coordinates = copy.deepcopy(route_1["geometry"]["coordinates"][0])
                except:
                    print(f"https://api.geoapify.com/v1/routing?waypoints={_from[1]}%2C{_from[0]}%7C{_to[1]}%2C{_to[0]}&mode=transit&apiKey={GEOAPIFY_KEY}")
                    coordinates = [_from, _to]
            else:
                r = requests.get(f"http://ors-app:8080/ors/v2/directions/driving-car?start={_from[0]},{ _from[1]}&end={_to[0]},{_to[1]}""")

                try:
                    routes = json.loads(r.content)
                    route_1 = routes.get("features")[0]
                    coordinates = copy.deepcopy(route_1["geometry"]["coordinates"])
                except:
                    print(f"http://ors-app:8080/ors/v2/directions/driving-car?start={_from[0]},{ _from[1]}&end={_to[0]},{_to[1]}")
                    coordinates = [_from, _to]

            points = []
 
            for index, coord in enumerate(coordinates):
                point = Point(coord[0], coord[1])
                points.append(point)
            
            aux = []
            for index in range(len(points)-1):
                aux.append([points[index], points[index+1]])
            
            for l in aux:
                lines.append([LineString(l)])

        self.gdf_lines2 = gpd.GeoDataFrame(lines, columns=['geometry'], crs=4326, geometry='geometry')
        #self.gdf_lines2.columns = [
        #    'geometry']
        #self.gdf_lines2.set_geometry('geometry')
        #self.gdf_lines2 = self.gdf_lines2[(
        #    np.abs(stats.zscore(self.gdf_lines2['consumo'])) < 3)]
        self.gdf_lines2.reset_index(inplace=True, drop=True)

    def plotMapa(self, input_path):
        m = folium.Map(max_bounds=True, tiles='CartoDB Positron')
        folium.GeoJson(self.gdf_lines2).add_to(m)
        sw = self.gdf[['Y', 'X']].min().values.tolist()
        ne = self.gdf[['Y', 'X']].max().values.tolist()

        m.fit_bounds([sw, ne])

        bus_stops = folium.FeatureGroup(name=f"Bus Stops").add_to(m)
        for index, row in self.data.iterrows():
            folium.Circle(location=(row['Y'], row['X']),
                          radius=2, fill=True).add_to(m)
            if row['Bus Stop'] == 1:
                marker = folium.Marker(
                        location=[row['Y'], row['X']],
                        icon=folium.Icon(icon='bus', prefix='fa')
                        )
                bus_stops.add_child(marker)
                
        folium.LayerControl().add_to(m)  
        filename = input_path.split('.')[0]
        m.save(f'maps/map_{filename}.html')
    
    
    def getMapa(self):
        m = folium.Map(max_bounds=True, tiles='CartoDB Positron')
        folium.TileLayer('openstreetmap').add_to(m)
        
        folium.GeoJson(self.gdf_lines2).add_to(m)
        sw = self.gdf[['Y', 'X']].min().values.tolist()
        ne = self.gdf[['Y', 'X']].max().values.tolist()

        m.fit_bounds([sw, ne])

        bus_stops = folium.FeatureGroup(name=f"Bus Stops").add_to(m)
        for index, row in self.data.iterrows():
            folium.Circle(location=(row['Y'], row['X']), color="black",
                          radius=2, fill=True, popup=folium.Popup(str(self.total_pairs[index]))).add_to(m)
            if row['Bus Stop'] == 1:
                marker = folium.Marker(
                        location=[row['Y'], row['X']],
                        icon=folium.Icon(icon='bus', prefix='fa'),
                        popup=folium.Popup(str(self.total_pairs[index]))
                        )
                bus_stops.add_child(marker)
                
        folium.LayerControl().add_to(m)  
        
        return m.get_root()._repr_html_()


    def pairsByDistance(self):
        pairs = []
        for i, row in self.gdf.iterrows():
            if i != 0:
                pairs.append((i - 1, i))
    
        return pairs
    
    
    def getLines_full(self, pairs=None):
        lines = []
        values = []
        self.all_desnivel = []

        for i in pairs:
            dist = self.gdf.at[i[1], 'distance']
            deltaH = self.gdf.at[i[1], 'Altitude'] - self.gdf.at[i[0], 'Altitude']
            desnivel = (deltaH * 100)/(dist)
            aux = []
            
            # Call the OSMR API
            _from = [self.data["X"][i[0]], self.data["Y"][i[0]]]
            _to = [self.data["X"][i[1]], self.data["Y"][i[1]]]

            if i[0] in self.exceptions_index:
                r = requests.get(f"https://api.geoapify.com/v1/routing?waypoints={_from[1]}%2C{_from[0]}%7C{_to[1]}%2C{_to[0]}&mode=bus&apiKey={GEOAPIFY_KEY}""")
                
                try:
                    routes = json.loads(r.content)
                    route_1 = routes.get("features")[0]
                    coordinates = copy.deepcopy(route_1["geometry"]["coordinates"][0])
                except:
                    print(f"https://api.geoapify.com/v1/routing?waypoints={_from[1]}%2C{_from[0]}%7C{_to[1]}%2C{_to[0]}&mode=bus&apiKey={GEOAPIFY_KEY}")
                    coordinates = [_from, _to]
            
            elif i[0] in self.transits_index:
                r = requests.get(f"https://api.geoapify.com/v1/routing?waypoints={_from[1]}%2C{_from[0]}%7C{_to[1]}%2C{_to[0]}&mode=transit&apiKey={GEOAPIFY_KEY}""")
                
                try:
                    routes = json.loads(r.content)
                    route_1 = routes.get("features")[0]
                    coordinates = copy.deepcopy(route_1["geometry"]["coordinates"][0])
                except:
                    print(f"https://api.geoapify.com/v1/routing?waypoints={_from[1]}%2C{_from[0]}%7C{_to[1]}%2C{_to[0]}&mode=transit&apiKey={GEOAPIFY_KEY}")
                    coordinates = [_from, _to]
            else:
                r = requests.get(f"http://ors-app:8080/ors/v2/directions/driving-car?start={_from[0]},{ _from[1]}&end={_to[0]},{_to[1]}""")

                try:
                    routes = json.loads(r.content)
                    route_1 = routes.get("features")[0]
                    coordinates = copy.deepcopy(route_1["geometry"]["coordinates"])
                except:
                    print(f"http://ors-app:8080/ors/v2/directions/driving-car?start={_from[0]},{ _from[1]}&end={_to[0]},{_to[1]}")
                    coordinates = [_from, _to]

            points = []

            for index, coord in enumerate(coordinates):
                point = Point(coord[0], coord[1])
                points.append(point)
            
            aux = []
            for index in range(len(points)-1):
                aux.append([points[index], points[index+1]])
            
            for l in aux:
                lines.append(LineString(l))
                values.append([desnivel])
            
            self.all_desnivel.append(desnivel)

        self.gdf_lines2 = gpd.GeoDataFrame(values, crs=4326, geometry=lines)
        self.gdf_lines2.columns = ['desnivel','geometry']
        self.gdf_lines2.reset_index(inplace=True, drop=True)
    
    def get_elevation_map(self, filename = '', map = "", bus_stop_icons = True):
        m = folium.Map(max_bounds=True, tiles='CartoDB Positron')
        folium.TileLayer('openstreetmap').add_to(m)

        self.colors = {}
        map_colors = {}

        for _, row in self.gdf_lines2.iterrows():
            feature = row['desnivel']
            norm = matplotlib.colors.Normalize(
                vmin=np.array(self.all_desnivel).min(), vmax=np.array(self.all_desnivel).max())
            cmap = matplotlib.colormaps['RdYlGn_r']
            rgba = cmap(norm(feature))
            self.colors[rgba] = feature
            map_colors[feature] = matplotlib.colors.rgb2hex(rgba)
        
        self.colors = {k: v for k, v in sorted(self.colors.items(), key=lambda item: item[1])}
        if len(self.colors.keys()) > 1:
            colormap = branca.colormap.LinearColormap(
                colors=self.colors.keys(), vmin=np.array(self.all_desnivel).min(), vmax=np.array(self.all_desnivel).max(),
                tick_labels= [np.array(self.all_desnivel).min(),
                            ((np.array(self.all_desnivel).max() + np.array(self.all_desnivel).min()) / 2 + np.array(self.all_desnivel).min()) / 2,
                            (np.array(self.all_desnivel).max() + np.array(self.all_desnivel).min()) / 2,
                            ((np.array(self.all_desnivel).max() + np.array(self.all_desnivel).min()) / 2 + np.array(self.all_desnivel).max()) / 2,
                                np.array(self.all_desnivel).max()])
            colormap.caption = 'Slope (%)'
            colormap.add_to(m)
        else:
            colormap = branca.colormap.LinearColormap(colors=[list(self.colors.keys())[0], 'red'],
                                                        vmin=np.array(self.all_desnivel).min(), vmax=1)
            colormap.caption = 'Slope (%)'
            colormap.add_to(m)

        folium.GeoJson(self.gdf_lines2,
                    style_function=lambda feature: {
                        'color': map_colors[feature['properties']['desnivel']],
                        'fillOpacity': 1,
                        'weight': 4
                        
                    }).add_to(m)
        sw = self.gdf[['Y', 'X']].min().values.tolist()
        ne = self.gdf[['Y', 'X']].max().values.tolist()

        m.fit_bounds([sw, ne])

        bus_stops = folium.FeatureGroup(name=f"Bus Stops").add_to(m)
        for index, row in self.data.iterrows():
            folium.Circle(location=(row['Y'], row['X']), color="black",
                          radius=2, fill=True).add_to(m)
            if row['Bus Stop'] == 1:
                marker = folium.Marker(
                        location=[row['Y'], row['X']],
                        icon=folium.Icon(icon='bus', prefix='fa')
                       
                        )
                bus_stops.add_child(marker)
        folium.LayerControl().add_to(m)      
        
        return m.get_root()._repr_html_()  
        


def get_html_map(route_name, mode, exception_indexes, transit_indexes):
    mapa = MapVisualization(f'src/middle_output_perfect/{route_name}', mode, exception_indexes, transit_indexes)
    mapa.setGeometry()
    pairs = mapa.pairsByDistance()
    mapa.getLines(pairs=pairs)
    return mapa.getMapa()


def get_elevation_map(route_name, mode, exception_indexes, transit_indexes):
    key = route_name.split('.')[0]
    mapa = MapVisualization(f'src/middle_output_perfect/{route_name}', mode, exception_indexes, transit_indexes, full=True, auxiliary=f'segmented_lines/processed_{key}_random_0.0%ze.csv')
    mapa.setGeometry()
    pairs = mapa.pairsByDistance()
    mapa.getLines_full(pairs=pairs)
    return mapa.get_elevation_map()

if __name__ == '__main__':

    mode = 2
    route_inputs = listdir(f'middle_output_perfect/')
    processed_files = listdir('maps/')
    for route_name in route_inputs:
        isprocessed = False
        for p in processed_files:
            if p.split('_')[2] == route_name.split('_')[0]:
                isprocessed = True
        do = False
        exceptions_index = []
        transits_index = []
        for key in ["22"]:#["H10", "H12", "V15", "V17", "7", "22"]:
            if key == route_name.split('_')[0]:
                do = True
                if key == "H12":
                    exceptions_index = [23, 24, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
                    transits_index = [38, 39, 40]
                if key == "7":
                    exceptions_index = [7, 8, 9, 10, 11, 47]
                    transits_index = [5, 6, 22, 23]   
        if do:#not isprocessed:
            mapa = MapVisualization(f'middle_output_perfect/{route_name}', mode, exceptions_index, transits_index)
            mapa.setGeometry()
            pairs = mapa.pairsByDistance()
            mapa.getLines(pairs=pairs)
            mapa.plotMapa(route_name)