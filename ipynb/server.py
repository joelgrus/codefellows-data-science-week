from flask import Flask, request

# you need to pip install flask-cors, or else you'll run into problems
from flask.ext.cors import CORS, cross_origin
import json
import math
import re
import urllib

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import pandas as pd

usage = pd.read_table('data/usage_2012.tsv',
                      parse_dates=['time_start', 'time_end'])
dates = usage.time_start.dt.date
starts = usage.station_start
df = pd.DataFrame({'dates':dates, 'starts':starts})
by_date_by_stop = df.pivot_table(index='dates',
                                 columns='starts',
                                 aggfunc=len,
                                 fill_value=0)
station_names = by_date_by_stop.columns

@app.route('/stations/')
@app.route('/stations/<fragment>/')
@cross_origin()
def search(fragment=""):
    relevant_stations = [station
                         for station in station_names
                         if re.search(fragment, station, re.I)]
    return json.dumps(relevant_stations)

@app.route('/station/<path:station>/')
@cross_origin()
def detail(station):
    print station
    station = urllib.unquote(station)
    print station
    if station not in by_date_by_stop:
        return ""
    series = by_date_by_stop[station]
    xs = [str(date)[:10] for date in series.index]
    ys = list(series)
    return json.dumps({ "x" : xs, "y" : ys})

if __name__ == '__main__':
    app.run()
