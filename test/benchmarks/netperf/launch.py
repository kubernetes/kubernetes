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

# Launch the netperf tests
#
# 1. Launch the netperf-orch service
# 2. Launch the worker pods
# 3. Wait for the output csv file to show up in GCS

import os
import jinja2
import subprocess
import time
import datetime
from optparse import OptionParser

CLOUD_STORAGE_FOLDER = "gs://gkalele-netperf-archive/"
DEBUG_LOG = "output.txt"
NAMESPACE = "netperf"

# Moved to public Docker hub
NETPERF_IMAGE        = "girishkalele/netperf-latest"

# Unique run id for storing datafiles
RUN_UUID = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

# Load jinja template for the netperf workers
POD_TEMPLATE = "netperf-rc.yaml.in"

parser = OptionParser()
parser.add_option("-n", "--iterations", dest="iterations", metavar="NUMBER", type="int", default=1,
                  help="Number of iterations to run")
parser.add_option("-H", "--hostnetworking", action="store_true", dest="hostnetworking", default=False,
                  help="(boolean) Enable Host Networking Mode for PODs")
parser.add_option("-t", "--tag", dest="tag", metavar="STRING", default=RUN_UUID,
                  help="Tag to use for this run and storage")
# This script has the ability to automatically push results to GCS storage buckets.
parser.add_option("-g", "--gcsstorage", action="store_true", dest="gcsstorage", default=False,
                  help="(boolean) Enable GCS storage for generated data and image files")
parser.add_option("-i", "--image", dest="netperf_image",
                  metavar="NETPERF_CONTAINER_IMAGE_PATH",
                  default=NETPERF_IMAGE,
                  help="Netperf container image to use for this run")

def cmd(cmdline, quiet=True):
  if quiet:
      cmdline = cmdline + ">/dev/null 2>&1"
  return os.system(cmdline)

def load_template():
  traw = open(POD_TEMPLATE).read()
  t = jinja2.Template(traw)
  return t

def get_node_names():
  # Scrape output of kubectl and get node names
  nodesText = subprocess.check_output(["kubectl", "get", "node"])
  nlines = nodesText.split("\n")[1:]
  nodes = []
  for line in nlines:
    node = line.split(" ")[0]
    if not node:
      continue
    if 'kubernetes-master' in node:
      continue # Cannot push pods to master
    nodes.append(node)
    print "Found kubenode '%s'" % node
  return nodes

def get_orchestrator_podname():
  # Scrape output of kubectl and get node names
  nodesText = subprocess.check_output(["kubectl", "get", "pod", "--no-headers", "--namespace", NAMESPACE])
  nlines = nodesText.split("\n")
  for line in nlines:
    podname = line.split(" ")[0]
    if not podname.startswith("netperf-orch-"):
      continue
    return podname
  return None

# Cleanup
def cleanup():
  for entity in [ "pod", "rc" ]:
    cmd("kubectl delete %s --all --namespace %s" % (entity, NAMESPACE))
  for svc in [ "netperf-w2", "netperf-orch" ]:
    cmd("kubectl delete svc %s --namespace %s" % (svc, NAMESPACE))

def create_services():
  print "-----------------------------------------------------------------------------------"
  cmd("kubectl create namespace %s" % NAMESPACE)
  print "Create a service for netperf-w2"
  cmd("kubectl create -f netperf-w2-svc.yaml", quiet=False)
  print "Launching orchestrator pod"
  cmd("kubectl create -f netperf-orch.yaml", quiet=False)

def create_pods(options, args, nodes):

  pod_template = load_template()

  for worker in range (1, 4):
    kubenode = nodes[1]
    if worker == 1 or worker == 2:
      kubenode = nodes[0]

    image = options.netperf_image

    # kubenode will be used for the nodeselector label
    data = {
        "podname": "netperf-w%d" % worker,
        "worker": "netperf-w%d" % worker,
        "image" : image,
        "mode"  : "worker",
        "kubenode" : kubenode,
        "hostnetworking" : ("%s"%options.hostnetworking).lower(),
    }

    # Render the 3 worker pod templates
    fd = open("netperf-w%d.yaml" % worker, "wt")
    fd.write(pod_template.render(data))
    fd.close()
    print "Launching worker pod netperf-w%d on node %s" % (worker, kubenode)
    cmd("kubectl create -f netperf-w%d.yaml" % worker, quiet=False)

def get_file_contents_from_pod(podname, filename):
    # Wrapper around "kubectl exec <pod> -- cat <filename>" and grab stdout
    data = ""
    try:
      data = subprocess.check_output(["kubectl", "exec", podname, "--namespace", NAMESPACE, "--", "cat", filename])
    except Exception, e:
      pass
    return data

def create_gcs_folder_path(options, args):
  if options.gcsstorage:
    FULLPATH = "%s%s" % (CLOUD_STORAGE_FOLDER, options.tag)
    return FULLPATH
  return ""

def pipeline_process_file(options, args, csvdata, iteration):
    # Once we have the raw CSV file from the orchestrator node
    fd = open(options.csv_file, "wt")
    fd.write(csvdata)
    fd.close()

    # Run the matplotlib helper to generate the graphs
    cmd("./plotperf.py --csv {0} --suffix {1}".format(options.csv_file, options.file_prefix)) # Generates svg, jpg and png plots

    # And dump them into a local directory and optionally into a GCS folder for archival
    GCS_PATH = create_gcs_folder_path(options, args)
    LOCAL_PATH = os.path.join(".", "data-{0}".format(options.tag))
    cmd("mkdir -p {0}".format(LOCAL_PATH))

    for ext in [ ".csv", ".png", ".svg", ".jpg", ".bar.png", ".bar.svg", ".bar.jpg" ]:
      src = "{0}{1}".format(options.file_prefix, ext)
      dst = "{0}/{1}-{2}{3}".format(LOCAL_PATH, options.file_prefix, iteration, ext)
      cmd("cp %s %s" % (src, dst))

      if options.gcsstorage:
        dst = "{0}/{1}-{2}{3}".format(GCS_PATH, options.file_prefix, iteration, ext)
        cmd("gsutil cp %s %s > /dev/null 2>&1" % (src, dst))

def execute(options, args):

  for iteration in range(0, options.iterations):

    cleanup()

    time.sleep(3)

    create_services()
    nodes = get_node_names()
    create_pods(options, args, nodes)

    print "Waiting for netperf pods to start up"
    time.sleep(60)

    # The pods orchestrate themselves, we just wait for the results file to show up in the orchestrator container
    while True:
      # Monitor the orchestrator pod for the CSV results file
      orchestrator_podname = get_orchestrator_podname()
      csvdata = get_file_contents_from_pod(orchestrator_podname, "/tmp/netperf.csv")

      if len(csvdata):
        print csvdata
        print "Results will also be written to %s" % options.csv_file
        pipeline_process_file(options, args, csvdata, iteration) # Push this iteration's data into the pipeline
        break

      print "Scanned orchestrator pod filesystem - no results file found yet...waiting for orchestrator to write CSV file..."
      time.sleep(60)

    # Cleanup resources
    print "TEST RUN (Iteration %d) FINISHED - cleaning up services and pods" % iteration

  # Finished requested iterations
  cleanup()

if __name__ == "__main__":

  (options, args) = parser.parse_args()
  options.file_prefix = "netperf-{0}".format(options.tag)
  options.csv_file = "{0}.csv".format(options.file_prefix)

  print "Network Performance Test"
  print "Parameters :"
  print "Iterations      : ", options.iterations
  print "Host Networking : ", options.hostnetworking
  print "------------------------------------------------------------"
  print "Testing kubectl functional"
  cmd("kubectl get node", quiet=False)

  print "kubectl functional - starting POD deployments"
  execute(options, args)
