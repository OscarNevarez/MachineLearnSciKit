from __future__ import print_function
from flask import Flask
from flask import Response
from flask import request
from sklearn import svm
import sys
import json


clf = svm.SVC(gamma=0.001, C=100.)
modelTrained = False
app = Flask(__name__)
data = []
@app.route('/')
def home():
    return app.send_static_file('front.html')


@app.route('/insertrecord/', methods=['GET'])
def insert_record():
    record = {"roadid" : int(request.args.get("roadid")), "direction" : int(request.args.get("direction")),
              "dayofweek" : int(request.args.get("dayofweek")), "timeofday" : int(request.args.get("timeofday")),
              "trafficstatus" : int(request.args.get("trafficstatus"))}

    data.append(record)
    print(str(len(data)) + " records in data", file=sys.stderr)
    return "success " + str(record["roadid"])+ " , " + str(record["direction"]) + " , " + str(record["dayofweek"]) + " , " + str(record["timeofday"]) + " , " + str(record["trafficstatus"]), 200


@app.route('/getprediction/', methods=['GET'])
def model_trainingdata():
    record = {"roadid": int(request.args.get("roadid")),
              "direction": int(request.args.get("direction")),
              "dayofweek": int(request.args.get("dayofweek")),
              "timeofday": int(request.args.get("timeofday"))}

    xvalues = getxvalues(data)
    print(xvalues, file=sys.stderr)
    yvalues = getyvalues(data)
    print(yvalues, file=sys.stderr)
    if(xvalues == 0 ):
        return "no training data was set, set data and try again"

    print(str(len(xvalues)) + " x values with " + str(len(yvalues)) + " y values ", file=sys.stderr)
    clf.fit(xvalues, yvalues)
    prediction = clf.predict([ [record["roadid"], record["direction"], record["dayofweek"], record["timeofday"] ]])
    dataResp = { 'prediction' : str(prediction)}
    js = json.dumps(dataResp)
    resp = Response(js, status = 200, mimetype='application/json')
    return resp


def getyvalues(data):
    yValues = []

    for record in data:
        yValues.append(record["trafficstatus"])

    return yValues


def getxvalues(data):
    xValues = []
    for record in data:
        xValues.append([record["roadid"], record["direction"], record["dayofweek"], record["timeofday"]])
    return xValues

if __name__ == '__main__':
    app.run(host='0.0.0.0')
