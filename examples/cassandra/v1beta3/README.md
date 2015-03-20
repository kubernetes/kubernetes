## Cloud Native Deployments of Cassandra using Kubernetes v1beta3 api

The following document describes the development of a _cloud native_ [Cassandra](http://cassandra.apache.org/) deployment on Kubernetes.  When we say _cloud native_ we mean an application which understands that it is running within a cluster manager, and uses this cluster management infrastructure to help implement the application.  In particular, in this instance, a custom Cassandra ```SeedProvider``` is used to enable Cassandra to dynamically discover new Cassandra nodes as they join the cluster.

This document also attempts to describe the core components of Kubernetes, _Pods_, _Services_ and _Replication Controllers_.

### Prerequisites
This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs/getting-started-guides) for installation instructions for your platform.


The v1beta3 API is not enabled by default. The kube-apiserver process needs to run with the --runtime_config=api/v1beta3 argument. Use the following command to enable it:
```sh
$sudo sed -i 's|KUBE_API_ARGS="|KUBE_API_ARGS="--runtime_config=api/v1beta3|' /etc/kubernetes/apiserver


```


### quickstart
For those of you who are impatient, here is the summary of the commands we ran in this tutorial.

```sh
# create a single cassandra node
kubectl create -f cassandra-controller.yaml

# create a service to track all cassandra nodes
kubectl create -f cassandra-service.yaml

$ docker exec <cassandra-container-id> nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address      Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.3.29  72.07 KB   256     100.0%            f736f0b5-bd1f-46f1-9b9d-7e8f22f37c9e  rack1

# scale up to 2 nodes
kubectl resize rc cassandra --replicas=2

# validate the cluster
$ docker exec <cassandra-container-id> nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address      Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.3.29  72.07 KB   256     100.0%            f736f0b5-bd1f-46f1-9b9d-7e8f22f37c9e  rack1
UN  10.244.1.10  41.14 KB   256     100.0%            42617acd-b16e-4ee3-9486-68a6743657b1  rack1

# scale up to 4 nodes
kubectl resize rc cassandra --replicas=4

# validate the cluster
$ docker exec <cassandra-container-id> nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address      Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.3.29  72.07 KB   256     49.5%             f736f0b5-bd1f-46f1-9b9d-7e8f22f37c9e  rack1
UN  10.244.2.14  61.62 KB   256     52.6%             3e9981a6-6919-42c4-b2b8-af50f23a68f2  rack1
UN  10.244.1.10  41.14 KB   256     49.5%             42617acd-b16e-4ee3-9486-68a6743657b1  rack1
UN  10.244.4.8   63.83 KB   256     48.3%             eeb73967-d1e6-43c1-bb54-512f8117d372  rack1
```
