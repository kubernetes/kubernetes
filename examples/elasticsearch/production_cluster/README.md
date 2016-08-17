<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/examples/elasticsearch/production_cluster/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Elasticsearch for Kubernetes

Kubernetes makes it trivial for anyone to easily build and scale [Elasticsearch](http://www.elasticsearch.org/) clusters. Here, you'll find how to do so.
Current Elasticsearch version is `1.7.1`.

Before we start, one needs to know that Elasticsearch best-practices recommend to separate nodes in three roles:
* `Master` nodes - intended for clustering management only, no data, no HTTP API
* `Client` nodes - intended for client usage, no data, with HTTP API
* `Data` nodes - intended for storing and indexing your data, no HTTP API

This is enforced throughout this document.

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING" width="25" height="25"> Current pod descriptors use an `emptyDir` for storing data in each data node container. This is meant to be for the sake of simplicity and [should be adapted according to your storage needs](../../../docs/design/persistent-storage.md).

## Docker image

This example uses [this pre-built image](https://github.com/pires/docker-elasticsearch-kubernetes) will not be supported. Feel free to fork to fit your own needs, but mind yourself that you will need to change Kubernetes descriptors accordingly.

## Deploy

```
kubectl create -f examples/elasticsearch/production_cluster/service-account.yaml
kubectl create -f examples/elasticsearch/production_cluster/es-discovery-svc.yaml
kubectl create -f examples/elasticsearch/production_cluster/es-svc.yaml
kubectl create -f examples/elasticsearch/production_cluster/es-master-rc.yaml
```

Wait until `es-master` is provisioned, and

```
kubectl create -f examples/elasticsearch/production_cluster/es-client-rc.yaml
```

Wait until `es-client` is provisioned, and

```
kubectl create -f examples/elasticsearch/production_cluster/es-data-rc.yaml
```

Wait until `es-data` is provisioned.

Now, I leave up to you how to validate the cluster, but a first step is to wait for containers to be in ```RUNNING``` state and check the Elasticsearch master logs:

```
$ kubectl get pods
NAME              READY     STATUS    RESTARTS   AGE
es-client-2ep9o   1/1       Running   0          2m
es-data-r9tgv     1/1       Running   0          1m
es-master-vxl6c   1/1       Running   0          6m
```

```
$ kubectl logs es-master-vxl6c
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
[2015-08-21 10:58:51,324][INFO ][node                     ] [Arc] version[1.7.1], pid[8], build[b88f43f/2015-07-29T09:54:16Z]
[2015-08-21 10:58:51,328][INFO ][node                     ] [Arc] initializing ...
[2015-08-21 10:58:51,542][INFO ][plugins                  ] [Arc] loaded [cloud-kubernetes], sites []
[2015-08-21 10:58:51,624][INFO ][env                      ] [Arc] using [1] data paths, mounts [[/data (/dev/sda9)]], net usable_space [14.4gb], net total_space [15.5gb], types [ext4]
[2015-08-21 10:58:57,439][INFO ][node                     ] [Arc] initialized
[2015-08-21 10:58:57,439][INFO ][node                     ] [Arc] starting ...
[2015-08-21 10:58:57,782][INFO ][transport                ] [Arc] bound_address {inet[/0:0:0:0:0:0:0:0:9300]}, publish_address {inet[/10.244.15.2:9300]}
[2015-08-21 10:58:57,847][INFO ][discovery                ] [Arc] myesdb/-x16XFUzTCC8xYqWoeEOYQ
[2015-08-21 10:59:05,167][INFO ][cluster.service          ] [Arc] new_master [Arc][-x16XFUzTCC8xYqWoeEOYQ][es-master-vxl6c][inet[/10.244.15.2:9300]]{data=false, master=true}, reason: zen-disco-join (elected_as_master)
[2015-08-21 10:59:05,202][INFO ][node                     ] [Arc] started
[2015-08-21 10:59:05,238][INFO ][gateway                  ] [Arc] recovered [0] indices into cluster_state
[2015-08-21 11:02:28,797][INFO ][cluster.service          ] [Arc] added {[Gideon][4EfhWSqaTqikbK4tI7bODA][es-data-r9tgv][inet[/10.244.59.4:9300]]{master=false},}, reason: zen-disco-receive(join from node[[Gideon][4EfhWSqaTqikbK4tI7bODA][es-data-r9tgv][inet[/10.244.59.4:9300]]{master=false}])
[2015-08-21 11:03:16,822][INFO ][cluster.service          ] [Arc] added {[Venomm][tFYxwgqGSpOejHLG4umRqg][es-client-2ep9o][inet[/10.244.53.2:9300]]{data=false, master=false},}, reason: zen-disco-receive(join from node[[Venomm][tFYxwgqGSpOejHLG4umRqg][es-client-2ep9o][inet[/10.244.53.2:9300]]{data=false, master=false}])
```

As you can assert, the cluster is up and running. Easy, wasn't it?

## Scale

Scaling each type of node to handle your cluster is as easy as:

```
kubectl scale --replicas=3 rc es-master
kubectl scale --replicas=2 rc es-client
kubectl scale --replicas=2 rc es-data
```

Did it work?

```
$ kubectl get pods
NAME              READY     STATUS    RESTARTS   AGE
es-client-2ep9o   1/1       Running   0          4m
es-client-ye5s1   1/1       Running   0          50s
es-data-8az22     1/1       Running   0          47s
es-data-r9tgv     1/1       Running   0          3m
es-master-57h7k   1/1       Running   0          52s
es-master-kuwse   1/1       Running   0          52s
es-master-vxl6c   1/1       Running   0          8m
```

Let's take another look of the Elasticsearch master logs:

```
$ kubectl logs es-master-vxl6c
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
[2015-08-21 10:58:51,324][INFO ][node                     ] [Arc] version[1.7.1], pid[8], build[b88f43f/2015-07-29T09:54:16Z]
[2015-08-21 10:58:51,328][INFO ][node                     ] [Arc] initializing ...
[2015-08-21 10:58:51,542][INFO ][plugins                  ] [Arc] loaded [cloud-kubernetes], sites []
[2015-08-21 10:58:51,624][INFO ][env                      ] [Arc] using [1] data paths, mounts [[/data (/dev/sda9)]], net usable_space [14.4gb], net total_space [15.5gb], types [ext4]
[2015-08-21 10:58:57,439][INFO ][node                     ] [Arc] initialized
[2015-08-21 10:58:57,439][INFO ][node                     ] [Arc] starting ...
[2015-08-21 10:58:57,782][INFO ][transport                ] [Arc] bound_address {inet[/0:0:0:0:0:0:0:0:9300]}, publish_address {inet[/10.244.15.2:9300]}
[2015-08-21 10:58:57,847][INFO ][discovery                ] [Arc] myesdb/-x16XFUzTCC8xYqWoeEOYQ
[2015-08-21 10:59:05,167][INFO ][cluster.service          ] [Arc] new_master [Arc][-x16XFUzTCC8xYqWoeEOYQ][es-master-vxl6c][inet[/10.244.15.2:9300]]{data=false, master=true}, reason: zen-disco-join (elected_as_master)
[2015-08-21 10:59:05,202][INFO ][node                     ] [Arc] started
[2015-08-21 10:59:05,238][INFO ][gateway                  ] [Arc] recovered [0] indices into cluster_state
[2015-08-21 11:02:28,797][INFO ][cluster.service          ] [Arc] added {[Gideon][4EfhWSqaTqikbK4tI7bODA][es-data-r9tgv][inet[/10.244.59.4:9300]]{master=false},}, reason: zen-disco-receive(join from node[[Gideon][4EfhWSqaTqikbK4tI7bODA][es-data-r9tgv][inet[/10.244.59.4:9300]]{master=false}])
[2015-08-21 11:03:16,822][INFO ][cluster.service          ] [Arc] added {[Venomm][tFYxwgqGSpOejHLG4umRqg][es-client-2ep9o][inet[/10.244.53.2:9300]]{data=false, master=false},}, reason: zen-disco-receive(join from node[[Venomm][tFYxwgqGSpOejHLG4umRqg][es-client-2ep9o][inet[/10.244.53.2:9300]]{data=false, master=false}])
[2015-08-21 11:04:40,781][INFO ][cluster.service          ] [Arc] added {[Erik Josten][QUJlahfLTi-MsxzM6_Da0g][es-master-kuwse][inet[/10.244.59.5:9300]]{data=false, master=true},}, reason: zen-disco-receive(join from node[[Erik Josten][QUJlahfLTi-MsxzM6_Da0g][es-master-kuwse][inet[/10.244.59.5:9300]]{data=false, master=true}])
[2015-08-21 11:04:41,076][INFO ][cluster.service          ] [Arc] added {[Power Princess][V4qnR-6jQOS5ovXQsPgo7g][es-master-57h7k][inet[/10.244.53.3:9300]]{data=false, master=true},}, reason: zen-disco-receive(join from node[[Power Princess][V4qnR-6jQOS5ovXQsPgo7g][es-master-57h7k][inet[/10.244.53.3:9300]]{data=false, master=true}])
[2015-08-21 11:04:53,966][INFO ][cluster.service          ] [Arc] added {[Cagliostro][Wpfx5fkBRiG2qCEWd8laaQ][es-client-ye5s1][inet[/10.244.15.3:9300]]{data=false, master=false},}, reason: zen-disco-receive(join from node[[Cagliostro][Wpfx5fkBRiG2qCEWd8laaQ][es-client-ye5s1][inet[/10.244.15.3:9300]]{data=false, master=false}])
[2015-08-21 11:04:56,803][INFO ][cluster.service          ] [Arc] added {[Thog][vkdEtX3ESfWmhXXf-Wi0_Q][es-data-8az22][inet[/10.244.15.4:9300]]{master=false},}, reason: zen-disco-receive(join from node[[Thog][vkdEtX3ESfWmhXXf-Wi0_Q][es-data-8az22][inet[/10.244.15.4:9300]]{master=false}])
```

## Access the service

*Don't forget* that services in Kubernetes are only acessible from containers in the cluster. For different behavior you should [configure the creation of an external load-balancer](http://kubernetes.io/v1.0/docs/user-guide/services.html#type-loadbalancer). While it's supported within this example service descriptor, its usage is out of scope of this document, for now.

```
$ kubectl get service elasticsearch
NAME            LABELS                                SELECTOR                              IP(S)          PORT(S)
elasticsearch   component=elasticsearch,role=client   component=elasticsearch,role=client   10.100.134.2   9200/TCP
```

From any host on your cluster (that's running `kube-proxy`), run:

```
curl http://10.100.134.2:9200
```

You should see something similar to the following:


```json
{
  "status" : 200,
  "name" : "Cagliostro",
  "cluster_name" : "myesdb",
  "version" : {
    "number" : "1.7.1",
    "build_hash" : "b88f43fc40b0bcd7f173a1f9ee2e97816de80b19",
    "build_timestamp" : "2015-07-29T09:54:16Z",
    "build_snapshot" : false,
    "lucene_version" : "4.10.4"
  },
  "tagline" : "You Know, for Search"
}
```

Or if you want to check cluster information:


```
curl http://10.100.134.2:9200/_cluster/health?pretty
```

You should see something similar to the following:

```json
{
  "cluster_name" : "myesdb",
  "status" : "green",
  "timed_out" : false,
  "number_of_nodes" : 7,
  "number_of_data_nodes" : 2,
  "active_primary_shards" : 0,
  "active_shards" : 0,
  "relocating_shards" : 0,
  "initializing_shards" : 0,
  "unassigned_shards" : 0,
  "delayed_unassigned_shards" : 0,
  "number_of_pending_tasks" : 0,
  "number_of_in_flight_fetch" : 0
}
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/elasticsearch/production_cluster/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
