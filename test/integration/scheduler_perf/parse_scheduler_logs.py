import sys
import datetime
import time
import math

pods = {}
nodes = {}
podA = {}

actualTestName = ""

def printDataSet():
    print("Processing " + actualTestName)
    keys = nodes[actualTestName].keys()
    keys.sort()
    i = 0
    sts = keys[0]

    f = open("{}.dat".format(actualTestName), "w")
    f.write("# " + actualTestName + "\n")
    for key in keys:
            tdiff = nodes[actualTestName][key].split(" ")[1]
            # tdiff = math.log(float(tdiff))
            f.write("{} {}\n".format((key-sts).total_seconds(), tdiff))
            i = i + 1
    f.close()

def printGnuPlotScript():
    f = open("{}.g".format(actualTestName), "w")
    f.write("set term png\n")
    f.write("set output \"" + actualTestName + ".png\"\n")
    f.write("set xlabel \"[total time]\"\n")
    f.write("set ylabel \"[pod scheduling duration (including retries)]\"\n")
    f.write("plot '{}.dat' using 1:2 title \"{}\" with lines\n".format(actualTestName, actualTestName.replace("_", " ")))
    f.close()

for line in sys.stdin.readlines():

    found = False
    for kL in ["scheduler.go", "scheduler_my_test.go"]:
        if kL in line:
            found = True
            break

    if not found:
        continue

    parts = filter(lambda x: len(x) > 0, line[:-1].split(" "))

    if "scheduler_my_test.go" in line:
        # New test
        if parts[4] == "Running":
            if actualTestName != "":
                printDataSet()
                printGnuPlotScript()

            actualTestName = parts[5][1:-1]
            pods = {}
            nodes = {}
            podA = {}

        continue

    # ts = time.mktime(datetime.datetime.strptime(parts[1], '%H:%M:%S.%f').timetuple())
    ts = datetime.datetime.strptime(parts[1], '%H:%M:%S.%f')
    if parts[4] == "Attempting":
        pods[parts[8]] = ts
        continue

    if parts[4] == "pod":
        pod = parts[5]
        try:
            nodes[actualTestName]
        except KeyError as e:
            nodes[actualTestName] = {}

        delta = (ts - pods[pod]).total_seconds()

        if pod not in podA:
            nodes[actualTestName][pods[pod]] = "{}: {}".format(pod, delta)
        podA[pod] = {}

# print the last data set
printDataSet()
printGnuPlotScript()
