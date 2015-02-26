## Cloud Native Deployments of Hazelcast using Kubernetes v1beta3 api

The following document describes the development of a _cloud native_ [Hazelcast](http://hazelcast.org/) deployment on Kubernetes.  When we say _cloud native_ we mean an application which understands that it is running within a cluster manager, and uses this cluster management infrastructure to help implement the application. In particular, in this instance, a custom Hazelcast ```bootstrapper``` is used to enable Hazelcast to dynamically discover Hazelcast nodes that have already joined the cluster.

Any topology changes are communicated and handled by Hazelcast nodes themselves.

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
# create a single hazelcast node
kubectl create -f hazelcast-controller.yaml

# create a service to track all hazelcast nodes
kubectl create -f hazelcast-service.yaml

# scale up to 2 nodes
kubectl resize rc hazelcast --replicas=2

# validate the cluster
docker exec <container-id> nodetool status

# scale up to 4 nodes
kubectl resize rc hazelcast --replicas=4

# Examine the status again by checking a node’s log and you should see the 4 members connected.
$ kubectl log 16b2beab-94a1-11e4-8a8b-42010af0e23e hazelcast
2014-12-24T01:21:09.731468790Z 2014-12-24 01:21:09.701  INFO 10 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Asking k8s registry at http://10.160.211.80:80..
2014-12-24T01:21:13.686978543Z 2014-12-24 01:21:13.686  INFO 10 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Found 3 pods running Hazelcast.
2014-12-24T01:21:13.772599736Z 2014-12-24 01:21:13.772  INFO 10 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Added member 10.160.2.3
2014-12-24T01:21:13.783689690Z 2014-12-24 01:21:13.783  INFO 10 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Added member 10.160.2.4

(...)

2014-12-24T01:21:16.007729519Z 2014-12-24 01:21:16.000  INFO 10 --- [cached.thread-3] c.h.nio.tcp.TcpIpConnectionManager       : [10.160.2.4]:5701 [someGroup] [3.3.3] Established socket connection between /10.160.2.4:54931 and /10.160.2.3:5701
2014-12-24T01:21:16.427289059Z 2014-12-24 01:21:16.427  INFO 10 --- [thread-Acceptor] com.hazelcast.nio.tcp.SocketAcceptor     : [10.160.2.4]:5701 [someGroup] [3.3.3] Accepting socket connection from /10.160.2.3:50660
2014-12-24T01:21:16.433763738Z 2014-12-24 01:21:16.433  INFO 10 --- [cached.thread-3] c.h.nio.tcp.TcpIpConnectionManager       : [10.160.2.4]:5701 [someGroup] [3.3.3] Established socket connection between /10.160.2.4:5701 and /10.160.2.3:50660
2014-12-24T01:21:23.036227250Z 2014-12-24 01:21:23.035  INFO 10 --- [ration.thread-1] com.hazelcast.cluster.ClusterService     : [10.160.2.4]:5701 [someGroup] [3.3.3]
2014-12-24T01:21:23.036227250Z
2014-12-24T01:21:23.036227250Z Members [3] {
2014-12-24T01:21:23.036227250Z  Member [10.160.2.4]:5701 this
2014-12-24T01:21:23.036227250Z  Member [10.160.2.3]:5701
2014-12-24T01:21:23.036227250Z }
```


