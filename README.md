# Bus Route Segmentation Service

## Install

Before starting the installation you must download the following compressed archive with the files with the elevation data and the Openrouteservice routing API:
[Large files]([https://docs.docker.com/engine/install/](https://drive.google.com/file/d/1WKKnH38WzV97rWczZRKWn4xAFzmn72xw/view?usp=share_link)) 


We suggest using docker to install and launch the backend of the Bus Line Segmentation Service. In short, a machine with a running [docker installation](https://docs.docker.com/engine/install/) will do everything for you. 

Before starting the docker container, you must configure its API KEY inside the Dockerfile by modifying line 32:
```
    ENV GEOAPIFY_KEY "your_api_key"
```

Also, you must include the files with the information about the bus lines under the routes directory.

To launch the docker container you can run the following command under the project directory:
```
    docker compose up -d
```

## Usage

Puede acceder al Servicio de Segmentación de Líneas de Bus en la siguiente URL:
[http://localhost:8082/](http://localhost:8082/)

If you want to add a bus line and you already have the docker container launched use the following command in the same directory where you have the data file:
```
    docker cp <filename> bus-line-segmentation-madrid:~/eMob_Madrid/src/middle_output_perfect/<filename>
```

To retrieve the segmented data from the service you can use the following command after the bus-line has been segmented:
```
    docker cp "bus-line-segmentation:eMob_Croatia/segmented_lines" ./
```

## Input format

The files with the information of the route to be segmented must be in *CSV* format, be located inside the *routes* directory and have the following columns:

* **From**: Coordinates of the initial stop of the segment in the following format:
    ```
    (latitude, longitude)
    ```
* **To**: Coordinates of the end stop of the segment in the following format:
    ```
    (latitude, longitude)
    ```
* **Final Stop**: 1 if it is the final stop before the bus route changes direction, otherwise 0.

