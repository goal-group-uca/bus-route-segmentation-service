import numpy as np
import pandas as pd
import math

from src.elevAPI import *

import matplotlib
import matplotlib.pyplot as plt
from pyowm import OWM
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from time import sleep
from shapely.geometry import Point, LineString
import xlrd
import geopandas as gpd

pd.options.mode.chained_assignment = None


def load_realroute(filename, model="srtm", lon_col="Longitude", lat_col="Latitude"):
    """Function to load a data file

    Parameters:
    df (Pandas DataFrame): Dataframe that contains Latitude-Longitude data of bus route
    model (string): Model used to calculate elevation using latitude-longitude data. options = {'srtm', 'merit', 'eudem', 'alos'}
    lon_col (string): Name of column which contains longitude coords
    lat_col (string): Name of column which contains latitude coords

    """

    ext = filename.split(".")[-1]
    if ext == "xlsx":
        data = pd.read_excel(filename, parse_dates=["Time"])
    elif ext == "csv":
        data = pd.read_csv(
            filename, parse_dates=["Time"], delimiter=None, sep=None, engine="python"
        )
    # data = pd.read_excel(filename, parse_dates=['Time'])
    # data = pd.read_excel(filename, parse_dates=['Time'], delimiter=None, sep=None)
    # data = pd.read_excel(filename, parse_dates=['Time'], delimiter=None)

    # data = pd.read_csv(filename, parse_dates=['Time'], delimiter=None, sep=None)

    # print(data.columns)
    # print(data.head())
    data["File"] = filename.split("/")[-1]

    # data['index_orig'] = data['index']

    data = propagateAxle(data)

    data.reset_index(inplace=True, drop=True)

    # data = rearrangeDataset(data)
    # data.reset_index(inplace=True, drop=True)

    elev = elevAPI(model)
    for i, row in data.iterrows():
        data.at[i, "Altitude"] = elev.getElevation(row[lat_col], row[lon_col])

    data = addColumns(data)

    ###########################
    ## Delete not ZE zones
    data = delete_Zones(data)
    data.reset_index(inplace=True, drop=True)
    ###########################

    aux = data.copy()
    # aux.index = aux['index']
    # aux.sort_index(inplace=True, ascending=False)
    aux.to_csv(
        "./output/Results_Formato_Original_"
        + filename.split("/")[-1].split(".")[0]
        + ".csv",
        sep=",",
    )

    # data = delete_ICE_Zones(data)
    # data.reset_index(inplace=True, drop=True)

    # data = checkChargeCycles(data)
    # data.reset_index(inplace=True, drop=True)

    data.drop(
        ["Event information", "Axle 3 Weight", "Axle 4 Weight", "VDOP", "HDOP"],
        axis=1,
        inplace=True,
    )

    pairs = pairsByZone(data)

    # pairs = pairsByDoors(data, pairs)

    pairs = list(map(tuple, dict(pairs).items()))

    for i in pairs:
        if data.at[i[1], "Odometer"] - data.at[i[0], "Odometer"] < 0.1:
            pairs.remove(i)

    # pairs = pairsBySlope(data, pairs)

    # pairs = list(map(tuple, dict(pairs).items()))

    pairs_orig = []
    for i in pairs:
        pairs_orig.append((data.at[i[0], "index"], data.at[i[1], "index"]))

    dict_pairs = dict(zip(pairs, pairs_orig))

    data = createNewDataset(data, pairs, dict_pairs)
    idx = data[(data["Dist"] < 0.1) & (data["Dist"] >= 0)].index
    data.drop(idx, axis=0, inplace=True)

    data["SoC ini"] = data["SoC ini"].astype(int)
    data["InitialSoc"] = data.iloc[0]["SoC ini"]
    return data


def propagateAxle(data):
    aux = data[data["Type"].str.contains("Doors closed")].iloc[0]
    ini = aux.name
    axle_1 = aux["Axle 1 Weight"]
    axle_2 = aux["Axle 2 Weight"]
    aux = data[
        (data["Type"] == "Zone: Entered") & (data["Description"].str.contains("ZE"))
    ]
    to_remove = None
    for i, _ in aux.iterrows():
        if i > ini:
            data.at[i, "Axle 1 Weight"] = axle_1
            data.at[i, "Axle 2 Weight"] = axle_2
            to_remove = np.arange(0, i)
            break
        else:
            pass
    # print(to_remove,'\n\n\n')
    # if to_remove!=None:
    data.drop(to_remove, axis=0, inplace=True)
    return data


def addColumns(data):
    axle1 = data.iloc[0]["Axle 1 Weight"]
    axle2 = data.iloc[0]["Axle 2 Weight"]

    data["Zone"] = None

    for i, _ in data.iterrows():
        if data.at[i, "Type"] == "Zone: Entered" or data.at[i, "Type"] == "Zone: Exit":
            aux = data.at[i, "Description"].split(" ,")
            data.at[i, "Zone"] = aux[0]
            if "SoC" in data.at[i, "Description"]:
                if len(aux) == 0:
                    data.at[i, "SoC"] = aux[0].split("SoC:")[-1].split(" %")[0].strip()
                elif len(aux) == 1:
                    data.at[i, "SoC"] = aux[0].split("SoC:")[-1].split(" %")[0].strip()
                else:
                    data.at[i, "SoC"] = aux[1].split("SoC:")[-1].split(" %")[0].strip()
            else:
                data.at[i, "SoC"] = data.at[i - 1, "SoC"]
        elif data.at[i, "Type"] == "Signal change":
            if "->" in data.at[i, "Description"]:
                data.at[i, "SoC"] = (
                    data.at[i, "Description"].split(" -> ")[-1].split(" %")[0].strip()
                )
            else:
                data.at[i, "SoC"] = (
                    data.at[i, "Description"].split("SoC: ")[1].split(" %")[0].strip()
                )
            if i != 0:
                data.at[i, "Zone"] = data.at[i - 1, "Zone"]

        else:
            if i > 0:
                data.at[i, "SoC"] = data.at[i - 1, "SoC"]
                data.at[i, "Zone"] = data.at[i - 1, "Zone"]
            else:
                data.at[i, "SoC"] = None
                data.at[i, "Zone"] = None

        if data.at[i, "Type"] == "Doors closed":
            axle1 = data.iloc[i]["Axle 1 Weight"]
            axle2 = data.iloc[i]["Axle 2 Weight"]
        else:
            data.at[i, "Axle 1 Weight"] = axle1
            data.at[i, "Axle 2 Weight"] = axle2
        if i != 0:
            dist = (data.at[i, "Odometer"] - data.at[i - 1, "Odometer"]) * 1000
            if dist != 0:
                data.at[i, "Pendiente"] = (
                    (data.at[i, "Altitude"] - data.at[i - 1, "Altitude"]) * 100
                ) / dist
            else:
                data.at[i, "Pendiente"] = data.at[i - 1, "Pendiente"]
        else:
            data.at[i, "Pendiente"] = 0
    return data


def delete_Zones(data):
    zone = data.loc[
        (data["Description"].str.contains("Zone: ZE") == False)
        & ((data["Type"] == "Zone: Entered") | (data["Type"] == "Zone: Exit"))
    ]
    for i, _ in zone.iterrows():
        zone.at[i, "Description"] = zone.at[i, "Description"].split(" ,")[0]

    lista = []
    for i in zone["Zone"].unique():
        A = data[(data["Type"] == "Zone: Entered") & (data["Zone"] == i)].index
        B = data[(data["Type"] == "Zone: Exit") & (data["Zone"] == i)].index
        lista.append(list(zip(A, B)))

    flatten = [item for sublist in lista for item in sublist]

    for i in flatten:
        idx = np.intersect1d(np.arange(i[0], i[1] + 1), data.index.to_numpy())
        data.drop(idx, axis=0, inplace=True)

    return data


def checkChargeCycles(data):
    lista = []
    for i, _ in data[data["Type"].str.contains("Vehicle stop") == True].iterrows():
        ciclo = False
        if (
            data.at[i + 1, "Type"] == "Signal change"
            and data.at[i + 2, "Type"] == "Signal change"
        ):
            for j, _ in data.iloc[i + 1 : :].iterrows():
                if (
                    data.iloc[j]["Type"] == "Signal change"
                    and data.iloc[j - 1]["SoC"] < data.iloc[j]["SoC"]
                ):
                    ciclo = True
                elif data.iloc[j]["Type"] == "Doors open":
                    if ciclo == True:
                        lista.append(np.arange(i, j + 3))
                        break
                else:
                    ciclo = False
    flatten = [item for sublist in lista for item in sublist]
    data.drop(flatten, axis=0, inplace=True)
    return data


def rearrangeDataset(data):
    aux = data.iloc[0]["Description"].split(" ,")
    zone = aux[0]
    closed = False
    for i, _ in data.iterrows():
        if "Zone:" in data.at[i, "Type"] and i != data.shape[0] and i != 0:
            if (
                data.at[i, "Description"].split(" ,")[0] == zone
                and data.at[i, "Type"] == "Zone: Exit"
                and closed == False
            ):
                closed = True

            if closed == True and data.at[i, "Type"] == "Zone: Entered":
                zone = data.at[i, "Description"].split(" ,")[0]
                closed = False

            if (
                data.at[i, "Type"] == "Zone: Entered"
                and data.at[i + 1, "Type"] == "Zone: Exit"
            ):
                if (
                    zone == data.at[i + 1, "Description"].split(" ,")[0]
                    and closed == False
                    and data.at[i + 1, "Description"].split(" ,")[0]
                    != data.at[i, "Description"].split(" ,")[0]
                ):
                    # print(i, i+1)
                    zone = data.at[i, "Description"].split(" ,")[0]
                    temp = data.iloc[i].copy()
                    data.iloc[i] = data.iloc[i + 1]
                    data.iloc[i + 1] = temp
                    closed = True
        else:
            closed = False
    return data


def pairsByZone(data):
    pairs = []
    for i, _ in data.iterrows():
        if data.at[i, "Type"] == "Zone: Entered":
            aux_zone = data.at[i, "Zone"]
            for j, _ in data[i::].iterrows():
                if (
                    data.at[j, "Type"] == "Zone: Exit"
                    and data.at[j, "Zone"] == aux_zone
                ):
                    pairs.append((i, j))
                    break
    return pairs


def pairsByDoors(data, pairs):
    axle1 = data.iloc[0]["Axle 1 Weight"]
    axle2 = data.iloc[0]["Axle 2 Weight"]

    counter = 0
    new_pairs = []
    for i in pairs:
        new_pairs.append(i)
        if i[0] != i[1] - 1:
            aux = data.loc[i[0] + 1 : i[1] + 1]
            counts = aux["Type"].value_counts()
            if (
                "Doors closed" in counts.index
            ):  # El bus ha abierto y cerrado puertas. Se corta el tramo
                idxs = aux[aux["Type"] == "Doors closed"].index
                counter_id = 0
                for idx in idxs:
                    if counter_id == 0:
                        pair = (i[0], idx)
                        del new_pairs[-1]
                        new_pairs.append(pair)
                    else:
                        new_pairs.append((idxs[counter_id - 1], idx))
                    counter_id += 1
    return new_pairs


def pairsBySlope(data, pairs):
    new_pairs = []
    counter = 0
    for i in pairs:
        new_pairs.append(i)
        idx = 0
        idx2 = 0
        if i[0] != i[1] - 1:
            counter_id = 0
            change = True
            aux_pair = i
            without_change = []
            for j, _ in data.iloc[i[0] : i[1]].iterrows():
                if j != i[1] + 1:
                    deltaPendiente = np.abs(
                        data.at[j + 1, "Pendiente"] - data.at[j, "Pendiente"]
                    )
                    sign = np.sign(data.at[j + 1, "Pendiente"])
                    if data.at[j + 1, "Pendiente"] == 0:
                        sign = 0
                    if sign != np.sign(data.at[j, "Pendiente"]):
                        if counter_id == 0:
                            del new_pairs[-1]
                        if change == True:
                            pair = (j, j + 1)
                        else:
                            if len(without_change) == 1:
                                pair = (without_change[0], j)
                            else:
                                pair = (without_change[0], without_change[-1])
                            new_pairs.append(pair)
                        new_pairs.append(pair)
                    else:
                        if deltaPendiente < 1:
                            without_change.append(j)
                            change = False
                        else:
                            change = True
                            if aux_pair == new_pairs[-1]:
                                del new_pairs[-1]
                            pair = (
                                j,
                                j + 1,
                            )
                            new_pairs.append(pair)
                else:
                    del new_pairs[-1]
                counter_id += 1
        counter += 0
    return new_pairs


def createNewDataset(data, pairs, dict_pairs):
    df = []

    counter = 0
    for index, i in enumerate(pairs):
        stops = 0
        # print(data.iloc[i[0]+1:i[1]+1])
        if i[0] != i[1] - 1:
            # print('entro\n')
            aux = data.iloc[i[0] + 1 : i[1] + 1]
            counts = aux["Type"].value_counts()
            if "Vehicle stop" in counts.index:
                stops = counts["Vehicle stop"]
        # print(data.at[i[0],'SoC'])
        if len(data.at[i[0], "SoC"]) == 0:
            soc_ini = "\0"
        else:
            soc_ini = int(data.at[i[0], "SoC"])
        if len(data.at[i[1], "SoC"]) == 0:
            soc_fin = "\0"
        elif soc_ini != "\0":
            soc_fin = int(data.at[i[1], "SoC"])
            deltasoc = int(soc_fin) - int(soc_ini)
        dist = data.at[i[1], "Odometer"] - data.at[i[0], "Odometer"]
        DeltaH = data.at[i[1], "Altitude"] - data.at[i[0], "Altitude"]
        pendiente = data.iloc[i[0] : i[1] + 1]["Pendiente"].mean()
        slope = math.atan(pendiente / 100)
        deltatime = data.at[i[1], "Time"] - data.at[i[0], "Time"]
        speed = (data.at[i[1], "Odometer"] - data.at[i[0], "Odometer"]) / (
            (deltatime.total_seconds() / 3600) + 1e-10
        )

        deltaspeed = data.at[i[1], "Speed"] - data.at[i[0], "Speed"]
        voltage = data.iloc[i[0] : i[1] + 1]["Battery Voltage"].mean()

        axle1 = data.at[i[0], "Axle 1 Weight"]
        axle2 = data.at[i[1], "Axle 2 Weight"]
        tramo = i
        tramo_orig = dict_pairs[i]
        recarga = np.sign(soc_fin - soc_ini)
        coolant_temp = data.iloc[i[0] : i[1] + 1]["Engine Coolant Temperature"].mean()

        acc = ((data.at[i[1], "Speed"] - data.at[i[0], "Speed"]) * 0.2777) / (
            deltatime.seconds + 1e-10
        )
        tramoCoords = LineString(
            [
                Point(data.at[i[0], "Longitude"], data.at[i[0], "Latitude"]),
                Point(data.at[i[1], "Longitude"], data.at[i[1], "Latitude"]),
            ]
        )
        # df.append(pd.Series([deltatime.seconds, dist, speed, deltaspeed, DeltaH, pendiente, voltage, axle1, axle2, soc_ini, soc_fin, deltasoc, stops, tramo, recarga, coolant_temp]))

        ###############################
        # Debug INFO
        # print("###################################")
        # print("{} -> {} : {}".format(data.at[i[0], 'Type'],type(data.at[i[0], 'Description']), data.at[i[0], 'Description']))
        # print("{} -> {} : {}".format(data.at[i[1], 'Type'],type(data.at[i[1], 'Description']), data.at[i[1], 'Description']))
        # print("###################################")

        zone_type = 0
        aux = index
        while type(data.at[pairs[aux][0], "Description"]) != str:
            aux -= 1

        if (
            "Zone: ZE" in data.at[pairs[aux][0], "Description"]
            and "Zone: Entered" in data.at[pairs[aux][0], "Type"]
        ):
            zone_type = 1

        df.append(
            pd.Series(
                [
                    deltatime.seconds,
                    dist,
                    speed,
                    deltaspeed,
                    acc,
                    DeltaH,
                    pendiente,
                    voltage,
                    axle1,
                    axle2,
                    soc_ini,
                    soc_fin,
                    deltasoc,
                    stops,
                    tramo,
                    tramo_orig,
                    tramoCoords,
                    recarga,
                    coolant_temp,
                    zone_type,
                    pendiente / 100,
                    slope,
                ]
            )
        )

    df = pd.DataFrame(df)
    # df.columns = ['Time', 'Dist', 'Velocidad', 'DeltaSpeed', 'DeltaH', '% Pendiente', 'Voltaje', 'Carga Eje 1', 'Carga Eje 2', 'SoC ini', 'SoC fin', 'DeltaSoC', 'Num Paradas', 'Tramo', 'Recarga', 'Temp_Refrigerante' ]
    df.columns = [
        "Time",
        "Dist",
        "Velocidad",
        "DeltaSpeed",
        "Aceleracion",
        "DeltaH",
        "% Pendiente",
        "Voltaje",
        "Carga Eje 1",
        "Carga Eje 2",
        "SoC ini",
        "SoC fin",
        "DeltaSoC",
        "Num Paradas",
        "Tramo",
        "Tramo_Orig",
        "Coordenadas",
        "Recarga",
        "Temp_Refrigerante",
        "Tipo de Zona",
        "Pendiente",
        "Ángulo de inclinación",
    ]

    geodf = gpd.GeoDataFrame(df.drop(["Tramo", "Tramo_Orig"], axis=1)).set_geometry(
        "Coordenadas"
    )
    geodf.to_file("Output.json", driver="GeoJSON")
    return df
