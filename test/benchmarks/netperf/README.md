# Benchmarking Kubernetes Networking Performance

## Objectives:
A standardized benchmark to measure Kubernetes networking performance on multiple host platforms and network stacks.

## Implementation and Test Methodology

The benchmark can be executed via a single Go binary invocation that triggers all the automated testing located in the orchestrator and worker pods as seen below. The test uses a custom docker container that has the go binary and iperf3 and other tools built into it. 
The orchestrator pod coordinates the worker pods to run tests in serial order for the 4 scenarios described below, at MTUs (MSS tuning for TCP and direct packet size tuning for UDP).
Using node labels, the Worker Pods 1 and 2 are placed on the same Kubernetes node, and Worker Pod 3 is placed on a different node. The nodes all communicate with the orchestrator pod service using simple golang rpcs and request work items. A minimum of two Kubernetes worker nodes are necessary for this test.

![](images/BenchmarkingKubernetesNetworkingPerformance.svg)

The 5 major network traffic paths are combination of Pod IP vs Virtual IP and whether the pods are co-located on the same node/VM versus a remotely located pod.

* Same VM using Pod IP

Same VM Pod to Pod traffic tests from Worker 1 to Worker 2 using its Pod IP.

* Same VM using Cluster/Virtual IP

Same VM Pod to Pod traffic tests from Worker 1 to Worker 2 using its Service IP (also known as its Cluster IP or Virtual IP).

* Remote VM using Pod IP

Worker 3 to Worker 2 traffic tests using Worker 2 Pod IP.

* Remote VM using Cluster/Virtual IP

Worker 3 to Worker 2 traffic tests using Worker 2 Cluster/Virtual IP.

* Same VM Pod Hairpin to itself using Cluster IP

The orchestrator and worker pods run independently of the initiator script, with the orchestrator pod sending work items to workers till the testcase schedule is complete.
The iperf output (both TPC and UDP modes) and the netperf TCP output from all worker nodes is uploaded to the orchestrator pod where it is filtered and the results are written to netperf.csv.
The launch script then extracts the netperf.csv file and writes it to local disk. All units in the csv file are in Gbits/second.
All Kubernetes entities are created under the “netperf” namespace.

Testcases can be added by extending the TestCase list in nptest/nptest.go

## Output Raw CSV data

```console
MSS                                          , Maximum, 96, 352, 608, 864, 1120, 1376,
1 iperf TCP. Same VM using Pod IP            ,35507.000000,33835,33430,35372,35220,35373,35507,
2 iperf TCP. Same VM using Virtual IP        ,32997.000000,32689,32997,32256,31995,31904,31830,
3 iperf TCP. Remote VM using Pod IP          ,10652.000000,8793,9836,10602,9959,9941,10652,
4 iperf TCP. Remote VM using Virtual IP      ,11046.000000,10429,11046,10064,10622,10528,10246,
5 iperf TCP. Hairpin Pod to own Virtual IP   ,32400.000000,31473,30253,32075,32058,32400,31734,
6 iperf UDP. Same VM using Pod IP            ,10642.000000,10642,
7 iperf UDP. Same VM using Virtual IP        ,8983.000000,8983,
8 iperf UDP. Remote VM using Pod IP          ,11143.000000,11143,
9 iperf UDP. Remote VM using Virtual IP      ,10836.000000,10836,
10 netperf. Same VM using Pod IP             ,11675.380000,11675.38,
11 netperf. Same VM using Virtual IP         ,0.000000,0.00,
12 netperf. Remote VM using Pod IP           ,6646.820000,6646.82,
13 netperf. Remote VM using Virtual IP       ,0.000000,0.00,

```

## Format of the benchmark results data

Ideally, we want the benchmark results to be consumable by regression e2e scripts as well as importable into a database for GCP vs GKE vs AWS for different VM sizes etc.
Currently, we have only iperf tests in the suite - the output for each MSS testpoint is a single value in Gbits/sec so the output is a simple CSV file with each column representing the transfer speed per MSS datapoint.

Q. Do we want to use CSV raw results format ? Alternative format suggestions ?

Q. Also calculate min and max for each of the 4 scenarios ?

## Graphing the CSV data

We generate a graph SVG file using matplotlib from the csv data (and also PNG and JPG files for compatibility).

* Python Matplotlib Line Charts

![](images/netperf.svg)

* Bar Charts 

Ignoring MSS, and using the Maximum bandwidth numbers, Bar Charts might be a better tool to compare performance:

![](images/netperf.bar.svg)
 
Importing the CSV data into Google sheets and graphing it also works.

## Optional: Data upload to GCS storage buckets
The docker container base image is google/cloud so it contains the gcloud and gsutil utilities. I have optional logic implemented so that we can upload the data directly from the orchestrator pod to GCS.
Or, we can have the launch python script upload to GCS.

# Test Modes and Pluggable CNI Modules

Command line parameters to the launch.go can switch the test mode to incorporate different podSpec settings.

* host networking

 The command line option * --hostnetworking * will inject the *hostNetwork: true* specifier into all the podSpec templates.
 
 ```bash
 
 $ go run ./launch.go -h
 Usage of launch.go:

  -hostnetworking
        (boolean) Enable Host Networking Mode for PODs
  -iterations int
        Number of iterations to run (default 1)

 $ go run ./launch.go --hostnetworking --iterations 1
 
 ```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/benchmarks/netperf/README.md?pixel)]()
