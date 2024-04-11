import math
import requests
import urllib
import json
import copy
import time
import random

import pandas as pd
import os
import tqdm

from src.elevAPI import elevAPI

os.environ["PROJ_LIB"] = (
        "/opt/conda/envs/gdal/share/proj"
)

GEOAPIFY_KEY = os.environ["GEOAPIFY_KEY"]

# Para configurar el porcentaje de zonas ZE a generar
config_ze_zones = [0.00]  # , 0.05, 0.10, 0.15, 1]
# Distancia mínima en metros para considerar un tramo
config_minimum_distance = 100
# Tiempo mínimo en segundos para considerar un tramo
config_minimum_time = 1
# Nombre del fichero de entrada
input_file = "parades_linia_Barcelona_conAlturas.xlsx"
# Carpeta en la que se encuentran los modelos de elevación a considerar
elevAPI_model = "src/CNIG"
# Columna donde está el nombre de la línea de bus:
NOMBRE_LINEA = "NOM_LINIA"
# Columna que indica el sentido de una linea:
SENTIDO_LINEA = "DESC_SENTIT"


def create_bus_stop_segments(data):
    new_data = []
    processed_routes = {}
    nom_linia = ""
    for i, row in data.iterrows():
        if i == 0:
            nom_linia = row[NOMBRE_LINEA]
        elif row[NOMBRE_LINEA] == prev_row[NOMBRE_LINEA] and row[SENTIDO_LINEA] != prev_row[SENTIDO_LINEA]:
            new_data[-1][2] = 1
        elif (i - 1) >= 0:
            if i == (len(data.index) - 1):
                prev_lon = prev_row["Longitud (X)"].replace("(", "")
                prev_lat = prev_row["Latitud (Y)"].replace(")", "")

                lon = row["Longitud (X)"].replace("(", "")
                lat = row["Latitud (Y)"].replace(")", "")

                aux = [
                    f"({prev_lat},{prev_lon})",  # From
                    f"({lat},{lon})",  # To
                    # prev_row["Altura"],  # Altitude From
                    # row["Altura"],  # Altitude To
                    0,  # Final Stop
                ]

                new_data.append(pd.Series(aux))

                new_data = pd.DataFrame(new_data)
                new_data.columns = [
                    "From",
                    "To",
                    # "Altitude From",
                    #   "Altitude To",
                    "Final Stop",
                ]

                processed_routes[nom_linia] = new_data

                new_data.to_csv(f"middle_outputs/{nom_linia}_{filename}.csv")

            elif row[NOMBRE_LINEA] == prev_row[NOMBRE_LINEA]:
                prev_lon = prev_row["Longitud (X)"].replace("(", "")
                prev_lat = prev_row["Latitud (Y)"].replace(")", "")

                lon = row["Longitud (X)"].replace("(", "")
                lat = row["Latitud (Y)"].replace(")", "")

                aux = [
                    f"({prev_lat},{prev_lon})",  # From
                    f"({lat},{lon})",  # To
                    # prev_row["Altura"],  # Altitude From
                    # row["Altura"],  # Altitude To
                    0,  # Final Stop
                ]

                new_data.append(pd.Series(aux))
            else:
                new_data = pd.DataFrame(new_data)
                new_data.columns = [
                    "From",
                    "To",
                    # "Altitude From",
                    #   "Altitude To",
                    "Final Stop",
                ]

                processed_routes[nom_linia] = new_data

                new_data.to_csv(f"middle_outputs/{nom_linia}_{filename}.csv")

                nom_linia = row[NOMBRE_LINEA]

                # New Bus Line
                new_data = []

        prev_row = row

    return processed_routes


def create_new_dataframe(input_file, elevAPI_model, exception_index, transit_index):
    # Init elevation model
    elev = elevAPI(elevAPI_model)
    filename = input_file.split("/")[-1].split(".")[0]

    # Create route segments
    routes_original = {}
    route_inputs = [input_file]
    for route_name in route_inputs:
        key = route_name.split('_')[0]
        data = pd.read_csv(f'src/middle_output_perfect/{route_name}', delimiter=None)
        routes_original[key] = data
    
    for key in routes_original.keys():
        isprocessed = False
        exceptions_index = []
        transits_index = []
            
        exceptions_index = [i+1 for i in exceptions_index]
        transits_index = [i+1 for i in transits_index]

        new_data = []
        data = routes_original[key]
        # Creating new dataframe
        for i, row in tqdm.tqdm(data.iterrows()):

            _from = eval(row["From"])
            _to = eval(row["To"])
            stop = True
            if_transit = False
            if_bus = False
                
            request_str = f"http://ors-app:8080/ors/v2/directions/driving-car?start={_from[1]},{ _from[0]}&end={_to[1]},{_to[0]}"
                
            if i in exceptions_index:
                request_str = f"https://api.geoapify.com/v1/routing?waypoints={_from[0]}%2C{_from[1]}%7C{_to[0]}%2C{_to[1]}&mode=bus&apiKey={GEOAPIFY_KEY}"
                if_bus = True
            elif i in transits_index:
                request_str = f"https://api.geoapify.com/v1/routing?waypoints={_from[0]}%2C{_from[1]}%7C{_to[0]}%2C{_to[1]}&mode=transit&apiKey={GEOAPIFY_KEY}"
                if_transit = True
                
            # Call the API
            r = requests.get(
                request_str
            )

            routes = json.loads(r.content)
            exception_detect = False
            try:
                route_1 = routes.get("features")[0]
                if if_transit or if_bus:
                    coordinates = copy.deepcopy(route_1["geometry"]["coordinates"][0])
                    segment_time = route_1["properties"]["time"]
                    segment_length = route_1["properties"]["distance"]
                    
                else:
                    coordinates = copy.deepcopy(route_1["geometry"]["coordinates"])
                    segment_time = route_1["properties"]['summary']['duration']
                    segment_length = route_1["properties"]['summary']['distance']
                    
            except:
                print(request_str)
                exception_detect = True

            # Departing Bus Stop -> Arriving Bus Stop
            begin_point = _from
            end_point = _to
            if not exception_detect:
                for index, coord in enumerate(coordinates):
                    if index < len(coordinates) - 2:
                        request_str = f"http://ors-app:8080/ors/v2/directions/driving-car?start={coordinates[index][0]},{coordinates[index][1]}&end={coordinates[index+1][0]},{coordinates[index+1][1]}"
                        
                        # Call the API
                        
                        if if_bus:
                            request_str = f"https://api.geoapify.com/v1/routing?waypoints={coordinates[index][1]}%2C{coordinates[index][0]}%7C{coordinates[index+1][1]}%2C{coordinates[index+1][0]}&mode=bus&apiKey={GEOAPIFY_KEY}"
                        elif if_transit:
                            request_str = f"https://api.geoapify.com/v1/routing?waypoints={coordinates[index][1]}%2C{coordinates[index][0]}%7C{coordinates[index+1][1]}%2C{coordinates[index+1][0]}&mode=transit&apiKey={GEOAPIFY_KEY}"

                        r = requests.get(
                            request_str
                        )
                        routes = json.loads(r.content)
                        route_1 = routes.get("features")[0]
                        try:
                            if if_bus or if_transit:
                                segment_time = route_1["properties"]["time"]
                                segment_length = route_1["properties"]["distance"]
                            else:
                                segment_time = route_1["properties"]['summary']['duration']
                                segment_length = route_1["properties"]['summary']['distance']

                            # Getting slope from section
                            a = elev.getElevation(
                                coordinates[index + 1][1], coordinates[index + 1][0]
                            )
                            b = elev.getElevation(coordinates[index][1], coordinates[index][0])

                            if a == None or b == None or segment_length == 0:
                                diferencia = 0
                                delta_slope = 0
                            else:
                                diferencia = a - b
                                delta_slope = diferencia / segment_length * 100

                            if index != 0:
                                if abs(previous_delta_slope - delta_slope) >= 2:
                                    # Call the API
                                    request_str = f"http://ors-app:8080/ors/v2/directions/driving-car?start={begin_point[1]},{begin_point[0]}&end={coordinates[index+1][0]},{coordinates[index+1][1]}"
                                    
                                    if if_bus:
                                        request_str = f"https://api.geoapify.com/v1/routing?waypoints={begin_point[0]}%2C{begin_point[1]}%7C{coordinates[index+1][1]}%2C{coordinates[index+1][0]}&mode=bus&apiKey={GEOAPIFY_KEY}"
                                    elif if_transit:
                                        request_str = f"https://api.geoapify.com/v1/routing?waypoints={begin_point[0]}%2C{begin_point[1]}%7C{coordinates[index+1][1]}%2C{coordinates[index+1][0]}&mode=transit&apiKey={GEOAPIFY_KEY}"
                                    
                                    r = requests.get(
                                        request_str
                                    )
                                    
                                    routes = json.loads(r.content)
                                    route_1 = routes.get("features")[0]
                                    
                                    request_str = f"http://ors-app:8080/ors/v2/directions/driving-car?start={coordinates[index+1][0]},{coordinates[index+1][1]}&end={end_point[1]},{end_point[0]}"

                                    if if_bus:
                                        request_str = f"https://api.geoapify.com/v1/routing?waypoints={coordinates[index+1][1]}%2C{coordinates[index+1][0]}%7C{end_point[0]}%2C{end_point[1]}&mode=bus&apiKey={GEOAPIFY_KEY}"
                                    elif if_transit:
                                        request_str = f"https://api.geoapify.com/v1/routing?waypoints={coordinates[index+1][1]}%2C{coordinates[index+1][0]}%7C{end_point[0]}%2C{end_point[1]}&mode=transit&apiKey={GEOAPIFY_KEY}"
                                    
                                    r = requests.get(
                                        request_str
                                    )
                                    routes = json.loads(r.content)
                                    route_2 = routes.get("features")[0]
                                    
                                    if if_bus or if_transit:
                                        segment_time_1 = route_1["properties"]["time"]
                                        segment_length_1 = route_1["properties"]["distance"]
                                        
                                        segment_time_2= route_2["properties"]["time"]
                                        segment_length_2 = route_2["properties"]["distance"]
                                    else:
                                        segment_time_1 = route_1["properties"]['summary']['duration']
                                        segment_length_1 = route_1["properties"]['summary']['distance']
                                        
                                        segment_time_2 = route_2["properties"]['summary']['duration']
                                        segment_length_2 = route_2["properties"]['summary']['distance']

                                    if (
                                        segment_length_1 > config_minimum_distance
                                        and segment_length_2 > config_minimum_distance
                                        and (
                                            segment_time_1 > config_minimum_time
                                            or not stop
                                        )
                                    ):
                                        # Section parameters
                                        duration = segment_time_1
                                        distance = segment_length_1
                                        avg_speed = distance / duration
                                        elev_from = elev.getElevation(
                                            begin_point[0], begin_point[1]
                                        )
                                        elev_to = elev.getElevation(
                                            coordinates[index + 1][1], coordinates[index + 1][0]
                                        )
                                        if elev_from == None or elev_to == None:
                                            diferencia = 0
                                            slope = 0
                                        else:
                                            diferencia = elev_to - elev_from
                                            slope = diferencia / distance
                                            
                                        slope_percent = slope * 100
                                        slope_angle = math.atan(slope)
                                        bus_stop = 1 if stop else 0
                                        final_stop = 0
                                        mid_point = copy.deepcopy(coordinates[index + 1])
                                        mid_point.reverse()
                                        mid_point = tuple(mid_point)

                                        new_data.append(
                                            pd.Series(
                                                [
                                                    begin_point,
                                                    mid_point,
                                                    duration,
                                                    distance,
                                                    avg_speed,
                                                    elev_from,
                                                    elev_to,
                                                    slope,
                                                    slope_percent,
                                                    slope_angle,
                                                    bus_stop,
                                                    final_stop,
                                                ]
                                            )
                                        )

                                        begin_point = mid_point
                                        stop = False

                            previous_delta_slope = delta_slope
                        except:
                            pass

                # Call the OSMR API
                request_str = f"http://ors-app:8080/ors/v2/directions/driving-car?start={begin_point[1]},{begin_point[0]}&end={end_point[1]},{end_point[0]}"
                
                if if_bus:
                    request_str = f"https://api.geoapify.com/v1/routing?waypoints={begin_point[0]}%2C{begin_point[1]}%7C{end_point[0]}%2C{end_point[1]}&mode=bus&apiKey={GEOAPIFY_KEY}"
                elif if_transit:
                    request_str = f"https://api.geoapify.com/v1/routing?waypoints={begin_point[0]}%2C{begin_point[1]}%7C{end_point[0]}%2C{end_point[1]}&mode=transit&apiKey={GEOAPIFY_KEY}"
                
                
                r = requests.get(
                    request_str
                )
                
                routes = json.loads(r.content)
                route_1 = routes.get("features")[0]
                
                if if_bus or if_transit:
                    segment_time = route_1["properties"]["time"]
                    segment_length = route_1["properties"]["distance"]
                else:
                    segment_time = route_1["properties"]['summary']['duration']
                    segment_length = route_1["properties"]['summary']['distance']
                    
                if segment_length > 0 and segment_time > 0:
                    # Section parameters
                    duration = segment_time
                    distance = segment_length
                    avg_speed = distance / duration
                    elev_from = elev.getElevation(begin_point[0], begin_point[1])
                    elev_to = elev.getElevation(end_point[0], end_point[1])
                    if elev_from == None or elev_to == None:
                        diferencia = 0
                        slope = 0
                    else:
                        diferencia = elev_to - elev_from
                        slope = diferencia / distance
                    slope_percent = slope * 100
                    slope_angle = math.atan(slope)
                    bus_stop = 1 if stop else 0
                    final_stop = row["Final Stop"]

                    new_data.append(
                        pd.Series(
                            [
                                begin_point,
                                end_point,
                                duration,
                                distance,
                                avg_speed,
                                elev_from,
                                elev_to,
                                slope,
                                slope_percent,
                                slope_angle,
                                bus_stop,
                                final_stop,
                            ]
                        )
                    )
            else:
                new_data.append(
                                pd.Series(
                                    [
                                        begin_point,
                                        end_point,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1 if stop else 0,
                                        row["Final Stop"],
                                    ]
                                )
                )
        new_data = pd.DataFrame(new_data)
        new_data.columns = [
            "From",
            "To",
            "Time",
            "Distance",
            "Avg Speed",
            "Altitude From",
            "Altitude To",
            "Slope",
            "Slope %",
            "Slope Angle",
            "Bus Stop",
            "Final Stop",
        ]
        
        #Check Elevations
        for index,row in new_data.iterrows():
            if row['Altitude To'] < -1000:
                before = row['Altitude From']
                if index != len(new_data) - 1:
                    after = new_data.iloc[index + 1]['Altitude To']

                    new_data.loc[index, 'Altitude To'] = (after + before) / 2
                    new_data.loc[index + 1, 'Altitude From'] = (after + before) / 2
                else:
                    new_data.loc[index, 'Altitude To'] = before

        new_data.to_csv(f"segmented_lines/processed_{filename}.csv")


def select_random_ze_zones(input_file, percentage, key):
    # Read input file
    data = pd.read_csv(input_file, index_col=0)

    # Selecting N random Zero Emissions Sections
    if percentage == 0.00:
        data["Zone Type"] = 0
    else:
        ze_indexes = random.sample(
            data.index.values.tolist(), round(len(data.index) * percentage)
        )

        data["Zone Type"] = 0
        data.loc[ze_indexes, "Zone Type"] = 1

    data.to_csv(
        "segmented_lines/processed_{}_random_{}%ze.csv".format(key, percentage * 100)
    )
    
    return data.to_html()

def get_segmented_route(input_file, exception_indexes, transit_indexes):
    elevAPI_model = "src/CNIG"
    create_new_dataframe(input_file, elevAPI_model, exception_indexes, transit_indexes)
    
    key = input_file.split('.')[0]
    return select_random_ze_zones("segmented_lines/processed_{}.csv".format(key), 0.00, key)

if __name__ == "__main__":
    random.seed(0)
    #create_new_dataframe(input_file, elevAPI_model)

    route_inputs = os.listdir(f'output/')
    for route_name in route_inputs:
        key = route_name.split('.')[0]
        for percentage in config_ze_zones:
            select_random_ze_zones("output/processed_{}.csv".format(key), percentage, key)
