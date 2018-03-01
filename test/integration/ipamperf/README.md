IPAM Performance Test
=====

Motivation
-----
We wanted to be able to test the behavior of the IPAM controller's under various scenarios, 
by mocking and monitoring the edges that the controller interacts with. This has the following goals:

- Save time on testing
- To simulate various behaviors cheaply
- To observe and model the ideal behavior of the IPAM controller code

Currently the test runs through the 4 different IPAM controller modes for cases where the kube API QPS is a) 
equal to and b) significantly less than the number of nodes being added to observe and quantify behavior.

How to run
-------

```shell
# In kubernetes root path
make generated_files

cd test/integration/ipamperf
./test-performance.sh
```

The runner scripts support a few different options:

```shell
./test-performance.sh -h
usage: ./test-performance.sh [-h] [-d] [-r <pattern>] [-o <filename>]
 -h display this help message
 -d enable debug logs in tests
 -r <pattern> regex pattern to match for tests
 -o <filename> file to write JSON formatted results to
```

The tests follow the pattern TestPerformance/{AllocatorType}-KubeQPS{X}-Nodes{Y}, where AllocatorType 
is one of 

- RangeAllocator
- IPAMFromCluster
- CloudAllocator
- IPAMFromCloud

and X represents the QPS configured for the kubernetes API client, and Y is the number of nodes to create.

The -d flags set the -v level for glog to 6, enabling nearly all of the debug logs in the code.

So to run the test for CloudAllocator with 10 nodes, one can run

```shell
./test-performance.sh -r /CloudAllocator.*Nodes10$
```

At the end of the test, a JSON format of the results for all the tests run is printed. Passing the -o option 
allows for also saving this JSON to a named file.

Code Organization
-----
The core of the tests are defined in [ipam_test.go](ipam_test.go), using the t.Run() helper to control parallelism
as we want to able to start the master once. [cloud.go](cloud.go) contains the mock of the cloud server endpoint
and can be configured to behave differently as needed by the various modes. The tracking of the node behavior and
creation of the test results data is in [results.go](results.go).
