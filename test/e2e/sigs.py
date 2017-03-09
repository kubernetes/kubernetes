#!/usr/bin/env python

# Copyright 2017 The Kubernetes Authors.
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



# This script attempts to match sigs to test/e2e/ files for auto/manual assignment when things break.
# Of course, its reasonable that sigs may reassign if the bot's guesses are not ideal.
# Its in the SIG's interest to maintain the 'sigs' data structure below.
# To maintain:
#   cd test/e2e ; python sigs.py > assignments.txt and commit the results.
# See https://github.com/kubernetes/kubernetes/issues/42670

import glob
import re
from sets import Set

go_files = glob.glob('*.go')

sigs = {
    "sig-api-machinery":["apiserver","admission","control","encoding","decoding"],
    "sig-storage":["etcd","cassandra","gluster","ceph","nfs","volume"],
    "sig-apps":["CLIs","SDK","package","downward","upward","Addon","job"],
    "sig-auth":["Authorization","Authentication","Security","SELinux","secrets"],
    "sig-autoscaling":["autoscaling","vertical", "horizontal", "heapster"],
    "sig-aws":["AWS", "Amazon", "S3"],
    "sig-big-data":["Spark", "Kafka", "Hadoop", "Flink", "Storm", "Cassandra","Mongo","Elasticsearch"],
    "sig-command-line-tools":["kubectl", "CLI"],
    "sig-federation":["federation"],
    "sig-cluster-lifecycle":["healthy","infrastructure","die","Disruptive"],
    "sig-network":["DNS", "Ingress", "Proxy","Traffic"],
    "sig-on-prem":["bare metal", "datacenter", "Baremetal"],
    "sig-openstack":["openstack", "Swift", "Neutron"],
    "sig-rkt":["rkt", "Rocket", "Coreos"],
    "sig-scheduling":["scheduling", "performance", "scale", "affinity", "density", "critical","taint","toleration","tolerate"],
    "sig-node":["docker","rkt","rocket","kubelet","taint","toleration","tolerate"],
    "sig-scale":["density","performance","load"],
    "sig-testing":["ginkgo", "infra", "conformance", "logging"],
    "sig-ui":["UX", "usability", "UI", "Web", "logging"],
    "sig-windows":["Windows"]
}
assignments = {}

for file_name in go_files:
    with open(file_name) as f:
        str = file_name + " = "
        if file_name not in assignments:
            assignments[file_name]=("no-sig",0)

        tags = Set([])
        scores = {
        }
        for s in sigs:
            scores[s]=0

        for l in f:
            for signal in ["It","By","should","the","per","create","Create","Describe","Sprint"]:
                if re.match('.*'+signal+'\(.*',l):
                    # process the words, lower weight but more information content
                    for sig in sigs:
                        for st in sigs[sig]:
                            result = re.search(st.upper(), l.upper())
                            if result:
                                if scores[sig]:
                                    scores[sig]=scores[sig]+2
                                else:
                                    scores[sig]=2;


                # process the tags, higher weight
                result = re.search('\[(.*)\]', l)
                if result:
                    for g in result.groups():
                        s = g
                        s = s.translate(None, '[')
                        s = s.translate(None, ']')
                        for t in s.split(' '):
                            tags.add(t)
                            for sig in sigs:
                                for st in sigs[sig]:
                                    if st.upper()==t.upper():
                                        if scores[sig]:
                                            scores[sig]=scores[sig]+5
                                        else:
                                            scores[sig]=5;

        for sig in scores:
            if assignments[file_name][1] < scores[sig]:
                assignments[file_name] = (sig,scores[sig])

print "**************** Sig assignments for tests ********************"

# print the main output: the assignments of test suites to sigs.
for a in assignments:
    if assignments[a][1]>0:
        print a, assignments[a]
    else:
        print a, assignments[a]," UNASSIGNED"

print "*************** Number of tests assigned per sig ***********************"

# Now output stats of how many assigned per sig.
counts = {}
for s in sigs:
    counts[s] = 0;
for s in sigs:
    for a in assignments:
        if assignments[a][0] == s :
            counts[s] = counts[s]+1
for sig in counts:
    print sig, counts[sig]