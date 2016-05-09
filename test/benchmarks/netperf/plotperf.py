#!/usr/bin/env python

# Copyright 2016 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Generates matplotlib line and bar charts from the netperf.csv raw data file
#

try:
  import matplotlib.pyplot as plt
except:
  print "Python package matplotlib not found - install with apt-get install python-matplotlib (or equivalent package manager)"
  raise

import numpy

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-c", "--csv", dest="csvfile", metavar="FILE",
                  help="Input CSV file")
parser.add_option("-s", "--suffix", dest="suffix",
                  help="Generated file suffix")
(options, args) = parser.parse_args()

def getData(filename):
  # Returns a list of lines split around the ',' and whitespace stripped
  fd = open(filename, "rt")
  lines = fd.readlines()
  fd.close()

  rl = []
  for l in lines:
    raw_elements = l.split(',')
    elements = [ e.strip() for e in raw_elements ]
    rl.append(elements)

  return rl

colors = [ 'r', 'g', 'b', 'c', 'm', 'k', '#ff6677' ]

def convert_float(l):
  rl = []
  for e in l:
    try:
      rl.append(float(e))
    except:
      pass
  return rl

if __name__ == "__main__":
  data = getData(options.csvfile)

  x_data = convert_float(data[0][2:])

  plt.figure(figsize=(16,6))
  plt.axis([0, 1500, 0, 45000])
  chart = plt.subplot(111)
  color_index = 0
  for n in range(1, len(data)):
    if len(data[n]) <= 4:
      continue
    y_dataset = convert_float(data[n][2:])
    chart.plot(x_data, y_dataset, marker=".", label=data[n][0], color=colors[color_index], linewidth=1.5)
    color_index += 1

  plt.xlabel("{0} - MSS or Packet Size".format(options.suffix))
  plt.ylabel("Mbits/sec")
  plt.title(options.suffix)

  # Shrink height by 10% on the bottom
  box = chart.get_position()
  chart.set_position([box.x0, box.y0,
                      box.width, box.height * 0.95])
  plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14,), ncol=3, borderaxespad=0.)

  for ext in [ "png", "svg", "jpg" ]:
    fname = "{0}.{1}".format(options.suffix, ext)
    plt.savefig(fname, dpi=100)
    print "Saved {0}".format(fname)

  barlabels = []
  barvalues = []
  for n in range(1, len(data)):
    l = l = data[n][0]
    splitOn='VM'
    l = ('\n%s'%splitOn).join(l.split(splitOn))
    barlabels.append(l)
    barvalues.append(float(data[n][1]))

  plt.clf()
  plt.barh(bottom=range(0, len(data)-1),
           height=0.5,
           width=barvalues,
           align='center')
  plt.yticks(numpy.arange(len(data)-1),
             barlabels)
  plt.grid(True)
  plt.title('Network Performance - Testcase {0}'.format(options.suffix))
  plt.xlabel("Testcase {0} - Mbits/sec".format(options.suffix))
  for ext in [ "png", "svg", "jpg" ]:
    fname = "{0}.bar.{1}".format(options.suffix, ext)
    plt.savefig(fname, dpi=100)
    print "Saved {0}".format(fname)
