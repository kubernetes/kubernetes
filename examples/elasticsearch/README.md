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
[here](http://releases.k8s.io/release-1.1/examples/elasticsearch/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Elasticsearch for Kubernetes

Kubernetes makes it trivial for anyone to easily build and scale [Elasticsearch](http://www.elasticsearch.org/) clusters. Here, you'll find how to do so.
Current Elasticsearch version is `2.0.0`.

[A more robust example that follows Elasticsearch best-practices of separating nodes concern is also available](production_cluster/README.md).

<img src="http://kubernetes.io/img/warning.png" alt="WARNING" width="25" height="25"> Current pod descriptors use an `emptyDir` for storing data in each data node container. This is meant to be for the sake of simplicity and [should be adapted according to your storage needs](../../docs/design/persistent-storage.md).

## Docker image

This example uses [this pre-built image](https://github.com/pires/docker-elasticsearch-kubernetes) will not be supported. Feel free to fork to fit your own needs, but mind yourself that you will need to change Kubernetes descriptors accordingly.

## Deploy

Let's kickstart our cluster with 1 instance of Elasticsearch.

```
kubectl create -f examples/elasticsearch/service-account.yaml
kubectl create -f examples/elasticsearch/es-svc.yaml
kubectl create -f examples/elasticsearch/es-rc.yaml
```

Let's see if it worked:

```
$ kubectl get pods
NAME                      READY     STATUS    RESTARTS   AGE
es-v8fzi                  1/1       Running   0          13s
```

```
$ kubectl logs es-v8fzi
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
[2015-11-20 18:27:53,449][INFO ][node                     ] [Danielle Moonstar] version[2.0.0], pid[13], build[de54438/2015-10-22T08:09:48Z]
[2015-11-20 18:27:53,456][INFO ][node                     ] [Danielle Moonstar] initializing ...
[2015-11-20 18:27:53,786][INFO ][plugins                  ] [Danielle Moonstar] loaded [cloud-kubernetes], sites []
[2015-11-20 18:27:53,823][INFO ][env                      ] [Danielle Moonstar] using [1] data paths, mounts [[/data (/dev/disk/by-uuid/7c2ba6f8-3e2f-49da-9d24-211659759bdb)]], net usable_space [90gb], net total_space [98.3gb], spins? [possibly], types [ext4]
[2015-11-20 18:27:56,545][INFO ][node                     ] [Danielle Moonstar] initialized
[2015-11-20 18:27:56,551][INFO ][node                     ] [Danielle Moonstar] starting ...
[2015-11-20 18:27:56,612][INFO ][transport                ] [Danielle Moonstar] publish_address {10.56.0.21:9300}, bound_addresses {10.56.0.21:9300}
[2015-11-20 18:27:56,622][INFO ][discovery                ] [Danielle Moonstar] myesdb/C9nmBJw3TJ22JcAVQmdNeg
[2015-11-20 18:28:01,571][INFO ][cluster.service          ] [Danielle Moonstar] new_master {Danielle Moonstar}{C9nmBJw3TJ22JcAVQmdNeg}{10.56.0.21}{10.56.0.21:9300}{master=true}, reason: zen-disco-join(elected_as_master, [0] joins received)
[2015-11-20 18:28:01,614][INFO ][http                     ] [Danielle Moonstar] publish_address {10.56.0.21:9200}, bound_addresses {10.56.0.21:9200}
[2015-11-20 18:28:01,615][INFO ][node                     ] [Danielle Moonstar] started
[2015-11-20 18:28:01,652][INFO ][gateway                  ] [Danielle Moonstar] recovered [0] indices into cluster_state
```

So we have a 1-node Elasticsearch cluster ready to handle some work.

## Scale

Please see [production_cluster](production_cluster/) for examples scaling Elasticsearch in Kubernetes.

## Access the service

*Don't forget* that services in Kubernetes are only acessible from containers in the cluster. For different behavior you should [configure the creation of an external load-balancer](http://kubernetes.io/v1.0/docs/user-guide/services.html#type-loadbalancer). While it's supported within this example service descriptor, its usage is out of scope of this document, for now.

```
$ kubectl get service elasticsearch
NAME            CLUSTER_IP      EXTERNAL_IP      PORT(S)             SELECTOR                  AGE
elasticsearch   10.59.241.197                    9200/TCP,9300/TCP   component=elasticsearch   14m
```

From any host on your cluster (that's running `kube-proxy`), run:

```
$ curl 10.59.241.197:9200
```

You should see something similar to the following:


```json
{
  "name" : "Yuriko Oyama",
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
curl 10.100.108.94:9200/_cluster/health?pretty
```

You should see something similar to the following:

```json

  "cluster_name" : "myesdb",
  "status" : "green",
  "timed_out" : false,
  "number_of_nodes" : 1,
  "number_of_data_nodes" : 1,
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
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/elasticsearch/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
