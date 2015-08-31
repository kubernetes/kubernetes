<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/examples/elasticsearch/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Elasticsearch for Kubernetes

This directory contains the source for a Docker image that creates an instance
of [Elasticsearch](https://www.elastic.co/products/elasticsearch) 1.5.2 which can
be used to automatically form clusters when used
with [replication controllers](../../docs/user-guide/replication-controller.md). This will not work with the library Elasticsearch image
because multicast discovery will not find the other pod IPs needed to form a cluster. This
image detects other Elasticsearch [pods](../../docs/user-guide/pods.md) running in a specified [namespace](../../docs/user-guide/namespaces.md) with a given
label selector. The detected instances are used to form a list of peer hosts which
are used as part of the unicast discovery mechanism for Elasticsearch. The detection
of the peer nodes is done by a program which communicates with the Kubernetes API
server to get a list of matching Elasticsearch pods.

Here is an example replication controller specification that creates 4 instances of Elasticsearch.

<!-- BEGIN MUNGE: EXAMPLE music-rc.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  labels:
    name: music-db
    namespace: mytunes
  name: music-db
spec:
  replicas: 4
  selector:
    name: music-db
  template:
    metadata:
      labels:
         name: music-db
    spec:
      containers:
      - name: es
        image: kubernetes/elasticsearch:1.2
        env:
          - name: "CLUSTER_NAME"
            value: "mytunes-db"
          - name: "SELECTOR"
            value: "name=music-db"
          - name: "NAMESPACE"
            value: "mytunes"
        ports:
        - name: es
          containerPort: 9200
        - name: es-transport
          containerPort: 9300
```

[Download example](music-rc.yaml)
<!-- END MUNGE: EXAMPLE music-rc.yaml -->

The `CLUSTER_NAME` variable gives a name to the cluster and allows multiple separate clusters to
exist in the same namespace.
The `SELECTOR` variable should be set to a label query that identifies the Elasticsearch
nodes that should participate in this cluster. For our example we specify `name=music-db` to
match all pods that have the label `name` set to the value `music-db`.
The `NAMESPACE` variable identifies the namespace
to be used to search for Elasticsearch pods and this should be the same as the namespace specified
for the replication controller (in this case `mytunes`).


Replace `NAMESPACE` with the actual namespace to be used. In this example we shall use
the namespace `mytunes`.

```yaml
kind: Namespace
apiVersion: v1
metadata:
  name: mytunes
  labels:
    name: mytunes
```

First, let's create the namespace:

```console
$ kubectl create -f examples/elasticsearch/mytunes-namespace.yaml 
namespaces/mytunes
```

Now you are ready to create the replication controller which will then create the pods:

```console
$ kubectl create -f examples/elasticsearch/music-rc.yaml --namespace=mytunes
replicationcontrollers/music-db
```

Let's check to see if the replication controller and pods are running:

```console
$ kubectl get rc,pods --namespace=mytunes
CONTROLLER   CONTAINER(S)   IMAGE(S)                       SELECTOR        REPLICAS
music-db     es             kubernetes/elasticsearch:1.2   name=music-db   4
NAME             READY     STATUS    RESTARTS   AGE
music-db-5p46b   1/1       Running   0          34s
music-db-8re0f   1/1       Running   0          34s
music-db-eq8j0   1/1       Running   0          34s
music-db-uq5px   1/1       Running   0          34s
```

It's also useful to have a [service](../../docs/user-guide/services.md) with an load balancer for accessing the Elasticsearch
cluster.

<!-- BEGIN MUNGE: EXAMPLE music-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  name: music-server
  namespace: mytunes
  labels:
    name: music-db
spec:
  selector:
    name: music-db
  ports:
  - name: db
    port: 9200
    targetPort: es
  type: LoadBalancer
```

[Download example](music-service.yaml)
<!-- END MUNGE: EXAMPLE music-service.yaml -->

Let's create the service with an external load balancer:

```console
$ kubectl create -f examples/elasticsearch/music-service.yaml --namespace=mytunes
services/music-server
```

Let's check the status of the service:

```console
$ kubectl get service --namespace=mytunes
NAME           LABELS          SELECTOR        IP(S)          PORT(S)
music-server   name=music-db   name=music-db   10.0.185.179   9200/TCP

```

Although this service has an IP address `10.0.185.179` internal to the cluster we don't yet have
an external IP address provisioned. Let's wait a bit and try again...

```console
$ kubectl get service --namespace=mytunes
NAME           LABELS          SELECTOR        IP(S)             PORT(S)
music-server   name=music-db   name=music-db   10.0.185.179      9200/TCP
                                               104.197.114.130 
```

Now we have an external IP address `104.197.114.130` available for accessing the service
from outside the cluster.

Let's see what we've got:

```console
$ kubectl get pods,rc,services --namespace=mytunes
NAME             READY     STATUS    RESTARTS   AGE
music-db-5p46b   1/1       Running   0          7m
music-db-8re0f   1/1       Running   0          7m
music-db-eq8j0   1/1       Running   0          7m
music-db-uq5px   1/1       Running   0          7m
CONTROLLER   CONTAINER(S)   IMAGE(S)                       SELECTOR        REPLICAS
music-db     es             kubernetes/elasticsearch:1.2   name=music-db   4
NAME           LABELS          SELECTOR        IP(S)             PORT(S)
music-server   name=music-db   name=music-db   10.0.185.179      9200/TCP
                                               104.197.114.130   
NAME                  TYPE                                  DATA
default-token-gcilu   kubernetes.io/service-account-token   2
```

This shows 4 instances of Elasticsearch running. After making sure that port 9200 is accessible for this cluster (e.g. using a firewall rule for Google Compute Engine) we can make queries via the service which will be fielded by the matching Elasticsearch pods.

```console
$ curl 104.197.114.130:9200
{
  "status" : 200,
  "name" : "Warpath",
  "cluster_name" : "mytunes-db",
  "version" : {
    "number" : "1.5.2",
    "build_hash" : "62ff9868b4c8a0c45860bebb259e21980778ab1c",
    "build_timestamp" : "2015-04-27T09:21:06Z",
    "build_snapshot" : false,
    "lucene_version" : "4.10.4"
  },
  "tagline" : "You Know, for Search"
}
$ curl 104.197.114.130:9200
{
  "status" : 200,
  "name" : "Callisto",
  "cluster_name" : "mytunes-db",
  "version" : {
    "number" : "1.5.2",
    "build_hash" : "62ff9868b4c8a0c45860bebb259e21980778ab1c",
    "build_timestamp" : "2015-04-27T09:21:06Z",
    "build_snapshot" : false,
    "lucene_version" : "4.10.4"
  },
  "tagline" : "You Know, for Search"
}
```

We can query the nodes to confirm that an Elasticsearch cluster has been formed.

```console
$ curl 104.197.114.130:9200/_nodes?pretty=true
{
  "cluster_name" : "mytunes-db",
  "nodes" : {
    "u-KrvywFQmyaH5BulSclsA" : {
      "name" : "Jonas Harrow",
...
        "discovery" : {
          "zen" : {
            "ping" : {
              "unicast" : {
                "hosts" : [ "10.244.2.48", "10.244.0.24", "10.244.3.31", "10.244.1.37" ]
              },
...
      "name" : "Warpath",
...
        "discovery" : {
          "zen" : {
            "ping" : {
              "unicast" : {
                "hosts" : [ "10.244.2.48", "10.244.0.24", "10.244.3.31", "10.244.1.37" ]
              },
...
        "name" : "Callisto",
...
        "discovery" : {
          "zen" : {
            "ping" : {
              "unicast" : {
                "hosts" : [ "10.244.2.48", "10.244.0.24", "10.244.3.31", "10.244.1.37" ]
              },
...
      "name" : "Vapor",
...
        "discovery" : {
          "zen" : {
            "ping" : {
              "unicast" : {
                "hosts" : [ "10.244.2.48", "10.244.0.24", "10.244.3.31", "10.244.1.37" ]
...
```

Let's ramp up the number of Elasticsearch nodes from 4 to 10:

```console
$ kubectl scale --replicas=10 replicationcontrollers music-db --namespace=mytunes
scaled
$ kubectl get pods --namespace=mytunes
NAME             READY     STATUS    RESTARTS   AGE
music-db-0n8rm   0/1       Running   0          9s
music-db-4izba   1/1       Running   0          9s
music-db-5dqes   0/1       Running   0          9s
music-db-5p46b   1/1       Running   0          10m
music-db-8re0f   1/1       Running   0          10m
music-db-eq8j0   1/1       Running   0          10m
music-db-p9ajw   0/1       Running   0          9s
music-db-p9u1k   1/1       Running   0          9s
music-db-rav1q   0/1       Running   0          9s
music-db-uq5px   1/1       Running   0          10m
```

Let's check to make sure that these 10 nodes are part of the same Elasticsearch cluster:

```console
$ curl 104.197.114.130:9200/_nodes?pretty=true | grep name
"cluster_name" : "mytunes-db",
      "name" : "Killraven",
        "name" : "Killraven",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Tefral the Surveyor",
        "name" : "Tefral the Surveyor",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Jonas Harrow",
        "name" : "Jonas Harrow",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Warpath",
        "name" : "Warpath",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Brute I",
        "name" : "Brute I",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Callisto",
        "name" : "Callisto",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Vapor",
        "name" : "Vapor",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Timeslip",
        "name" : "Timeslip",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Magik",
        "name" : "Magik",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Brother Voodoo",
        "name" : "Brother Voodoo",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/elasticsearch/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->