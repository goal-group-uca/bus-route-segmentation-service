from bottle import route, run, template, static_file, request

from src.coord_map import get_html_map, get_elevation_map
from src.processBusRouteV2 import get_segmented_route

@route('/', method='GET')
def index():
    return static_file('index.html', root='templates/')

@route('/resources/<filepath:path>', method='GET')
def resources(filepath):
    return static_file(filepath, root='resources/' )

@route('/segmented_lines/<filepath:path>', method='GET')
def csvs(filepath):
    return static_file(filepath, root='segmented_lines/' )

@route('/css/<filepath:path>', method='GET')
def css(filepath):
    return static_file(filepath, root='css/' )

@route('/postInputBusLine', method='POST')
def post_input_busline():
    return {
        "STATUS": "OK",
        "map": get_html_map(request.json['filename'], request.json['busdirection'], request.json['exceptionIndexes'], request.json['transitIndexes'])
        }

@route('/segmentBusLine', method='POST')
def post_input_busline():
    return {
        "STATUS": "OK",
        "map": get_segmented_route(request.json['filename'], request.json['exceptionIndexes'], request.json['transitIndexes'])
        }
    
@route('/elevationBusLine', method='POST')
def post_input_busline():
    return {
        "STATUS": "OK",
        "map": get_elevation_map(request.json['filename'], request.json['busdirection'], request.json['exceptionIndexes'], request.json['transitIndexes'])
        }


if __name__ == '__main__':
    run(host='0.0.0.0', port=8082, reloader=True)