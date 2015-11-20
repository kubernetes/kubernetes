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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/examples/elasticsearch/production_cluster/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Elasticsearch for Kubernetes

Kubernetes makes it trivial for anyone to easily build and scale [Elasticsearch](http://www.elasticsearch.org/) clusters. Here, you'll find how to do so.
Current Elasticsearch version is `2.0.0`.

Before we start, one needs to know that Elasticsearch best-practices recommend to separate nodes in three roles:
* `Master` nodes - intended for clustering management only, no data, no HTTP API
* `Client` nodes - intended for client usage, no data, with HTTP API
* `Data` nodes - intended for storing and indexing your data, no HTTP API

This is enforced throughout this document.

<img src="http://kubernetes.io/img/warning.png" alt="WARNING" width="25" height="25"> Current pod descriptors use an `emptyDir` for storing data in each data node container. This is meant to be for the sake of simplicity and [should be adapted according to your storage needs](../../../docs/design/persistent-storage.md).

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
NAME                      READY     STATUS    RESTARTS   AGE
es-client-foqvh           1/1       Running   0          27s
es-data-5qo1f             1/1       Running   0          14s
es-master-iwwzq           1/1       Running   0          2m
```

```
$ kubectl logs es-master-iwwzq
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
[2015-11-20 17:11:16,586][INFO ][node                     ] [Dragon Lord] version[2.0.0], pid[13], build[de54438/2015-10-22T08:09:48Z]
[2015-11-20 17:11:16,586][INFO ][node                     ] [Dragon Lord] initializing ...
[2015-11-20 17:11:16,938][INFO ][plugins                  ] [Dragon Lord] loaded [cloud-kubernetes], sites []
[2015-11-20 17:11:16,973][INFO ][env                      ] [Dragon Lord] using [1] data paths, mounts [[/data (/dev/disk/by-uuid/7c2ba6f8-3e2f-49da-9d24-211659759bdb)]], net usable_space [90.1gb], net total_space [98.3gb], spins? [possibly], types [ext4]
[2015-11-20 17:11:19,545][INFO ][node                     ] [Dragon Lord] initialized
[2015-11-20 17:11:19,545][INFO ][node                     ] [Dragon Lord] starting ...
[2015-11-20 17:11:19,614][INFO ][transport                ] [Dragon Lord] publish_address {10.56.0.18:9300}, bound_addresses {10.56.0.18:9300}
[2015-11-20 17:11:19,622][INFO ][discovery                ] [Dragon Lord] myesdb/7j08TLoRRDa5pnZiGfN9sA
[2015-11-20 17:11:23,881][INFO ][cluster.service          ] [Dragon Lord] new_master {Dragon Lord}{7j08TLoRRDa5pnZiGfN9sA}{10.56.0.18}{10.56.0.18:9300}{data=false, master=true}, reason: zen-disco-join(elected_as_master, [0] joins received)
[2015-11-20 17:11:23,898][INFO ][node                     ] [Dragon Lord] started
[2015-11-20 17:11:23,935][INFO ][gateway                  ] [Dragon Lord] recovered [0] indices into cluster_state
[2015-11-20 17:13:16,171][INFO ][cluster.service          ] [Dragon Lord] added {{Gog}{R2t5vWTRTYCjwdPWAvOt7A}{10.56.0.19}{10.56.0.19:9300}{data=false, master=false},}, reason: zen-disco-join(join from node[{Gog}{R2t5vWTRTYCjwdPWAvOt7A}{10.56.0.19}{10.56.0.19:9300}{data=false, master=false}])
[2015-11-20 17:13:30,052][INFO ][cluster.service          ] [Dragon Lord] added {{Stryfe}{lzrKBd4RTxuBY5SxcQOstg}{10.56.0.20}{10.56.0.20:9300}{master=false},}, reason: zen-disco-join(join from node[{Stryfe}{lzrKBd4RTxuBY5SxcQOstg}{10.56.0.20}{10.56.0.20:9300}{master=false}])
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
NAME                      READY     STATUS    RESTARTS   AGE
es-client-foqvh           1/1       Running   0          2m
es-client-oqfjk           1/1       Running   0          14s
es-data-5qo1f             1/1       Running   0          2m
es-data-75pic             1/1       Running   0          6s
es-master-00e41           1/1       Running   0          24s
es-master-4elbk           1/1       Running   0          24s
es-master-iwwzq           1/1       Running   0          4m
```

Let's take another look of the Elasticsearch master logs:

```
$ kubectl logs es-master-iwwzq
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
[2015-11-20 17:11:16,586][INFO ][node                     ] [Dragon Lord] version[2.0.0], pid[13], build[de54438/2015-10-22T08:09:48Z]
[2015-11-20 17:11:16,586][INFO ][node                     ] [Dragon Lord] initializing ...
[2015-11-20 17:11:16,938][INFO ][plugins                  ] [Dragon Lord] loaded [cloud-kubernetes], sites []
[2015-11-20 17:11:16,973][INFO ][env                      ] [Dragon Lord] using [1] data paths, mounts [[/data (/dev/disk/by-uuid/7c2ba6f8-3e2f-49da-9d24-211659759bdb)]], net usable_space [90.1gb], net total_space [98.3gb], spins? [possibly], types [ext4]
[2015-11-20 17:11:19,545][INFO ][node                     ] [Dragon Lord] initialized
[2015-11-20 17:11:19,545][INFO ][node                     ] [Dragon Lord] starting ...
[2015-11-20 17:11:19,614][INFO ][transport                ] [Dragon Lord] publish_address {10.56.0.18:9300}, bound_addresses {10.56.0.18:9300}
[2015-11-20 17:11:19,622][INFO ][discovery                ] [Dragon Lord] myesdb/7j08TLoRRDa5pnZiGfN9sA
[2015-11-20 17:11:23,881][INFO ][cluster.service          ] [Dragon Lord] new_master {Dragon Lord}{7j08TLoRRDa5pnZiGfN9sA}{10.56.0.18}{10.56.0.18:9300}{data=false, master=true}, reason: zen-disco-join(elected_as_master, [0] joins received)
[2015-11-20 17:11:23,898][INFO ][node                     ] [Dragon Lord] started
[2015-11-20 17:11:23,935][INFO ][gateway                  ] [Dragon Lord] recovered [0] indices into cluster_state
[2015-11-20 17:13:16,171][INFO ][cluster.service          ] [Dragon Lord] added {{Gog}{R2t5vWTRTYCjwdPWAvOt7A}{10.56.0.19}{10.56.0.19:9300}{data=false, master=false},}, reason: zen-disco-join(join from node[{Gog}{R2t5vWTRTYCjwdPWAvOt7A}{10.56.0.19}{10.56.0.19:9300}{data=false, master=false}])
[2015-11-20 17:13:30,052][INFO ][cluster.service          ] [Dragon Lord] added {{Stryfe}{lzrKBd4RTxuBY5SxcQOstg}{10.56.0.20}{10.56.0.20:9300}{master=false},}, reason: zen-disco-join(join from node[{Stryfe}{lzrKBd4RTxuBY5SxcQOstg}{10.56.0.20}{10.56.0.20:9300}{master=false}])
[2015-11-20 17:15:39,365][INFO ][cluster.service          ] [Dragon Lord] added {{Bird-Brain}{ZlwQuV8fSWqvbeU6vcvnQw}{10.56.2.12}{10.56.2.12:9300}{data=false, master=true},}, reason: zen-disco-join(join from node[{Bird-Brain}{ZlwQuV8fSWqvbeU6vcvnQw}{10.56.2.12}{10.56.2.12:9300}{data=false, master=true}])
[2015-11-20 17:15:39,873][INFO ][cluster.service          ] [Dragon Lord] added {{Captain Barracuda}{PMJ1PTLYR5i8NGShUfHZqA}{10.56.1.10}{10.56.1.10:9300}{data=false, master=true},}, reason: zen-disco-join(join from node[{Captain Barracuda}{PMJ1PTLYR5i8NGShUfHZqA}{10.56.1.10}{10.56.1.10:9300}{data=false, master=true}])
[2015-11-20 17:15:49,514][INFO ][cluster.service          ] [Dragon Lord] added {{Windshear}{cZ7n8MPdQDKDR4vaq97p8Q}{10.56.2.13}{10.56.2.13:9300}{data=false, master=false},}, reason: zen-disco-join(join from node[{Windshear}{cZ7n8MPdQDKDR4vaq97p8Q}{10.56.2.13}{10.56.2.13:9300}{data=false, master=false}])
[2015-11-20 17:15:58,818][INFO ][cluster.service          ] [Dragon Lord] added {{Spirit of '76}{ckJzSSd6T0-0p2rDU1cIew}{10.56.1.11}{10.56.1.11:9300}{master=false},}, reason: zen-disco-join(join from node[{Spirit of '76}{ckJzSSd6T0-0p2rDU1cIew}{10.56.1.11}{10.56.1.11:9300}{master=false}])
```

## Access the service

*Don't forget* that services in Kubernetes are only acessible from containers in the cluster. For different behavior you should [configure the creation of an external load-balancer](http://kubernetes.io/v1.0/docs/user-guide/services.html#type-loadbalancer). While it's supported within this example service descriptor, its usage is out of scope of this document, for now.

```
$ kubectl get service elasticsearch
NAME            CLUSTER_IP     EXTERNAL_IP      PORT(S)    SELECTOR                              AGE
elasticsearch   10.59.250.92                    9200/TCP   component=elasticsearch,role=client   5m
```

From any host on your cluster (that's running `kube-proxy`), run:

```
curl http://10.59.250.92:9200
```

You should see something similar to the following:


```json
{
  "name" : "Gog",
  "cluster_name" : "myesdb",
  "version" : {
    "number" : "2.0.0",
    "build_hash" : "de54438d6af8f9340d50c5c786151783ce7d6be5",
    "build_timestamp" : "2015-10-22T08:09:48Z",
    "build_snapshot" : false,
    "lucene_version" : "5.2.1"
  },
  "tagline" : "You Know, for Search"
}
```

Or if you want to check cluster information:


```
curl http://10.59.250.92:9200/_cluster/health?pretty
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
  "number_of_in_flight_fetch" : 0,
  "task_max_waiting_in_queue_millis" : 0,
  "active_shards_percent_as_number" : 100.0
}
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/elasticsearch/production_cluster/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
