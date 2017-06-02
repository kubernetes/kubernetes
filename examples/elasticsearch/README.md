# Elasticsearch for Kubernetes

Kubernetes makes it trivial for anyone to easily build and scale [Elasticsearch](http://www.elasticsearch.org/) clusters. Here, you'll find how to do so.
Current Elasticsearch version is `1.7.1`.

[A more robust example that follows Elasticsearch best-practices of separating nodes concern is also available](production_cluster/README.md).

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING" width="25" height="25"> Current pod descriptors use an `emptyDir` for storing data in each data node container. This is meant to be for the sake of simplicity and [should be adapted according to your storage needs](https://kubernetes.io/docs/design/persistent-storage.md).

## Docker image

The [pre-built image](https://github.com/pires/docker-elasticsearch-kubernetes) used in this example will not be supported. Feel free to fork to fit your own needs, but keep in mind that you will need to change Kubernetes descriptors accordingly.

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
NAME             READY     STATUS    RESTARTS   AGE
es-kfymw         1/1       Running   0          7m
kube-dns-p3v1u   3/3       Running   0          19m
```

```
$ kubectl logs es-kfymw
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
[2015-08-30 10:01:31,946][INFO ][node                     ] [Hammerhead] version[1.7.1], pid[7], build[b88f43f/2015-07-29T09:54:16Z]
[2015-08-30 10:01:31,946][INFO ][node                     ] [Hammerhead] initializing ...
[2015-08-30 10:01:32,110][INFO ][plugins                  ] [Hammerhead] loaded [cloud-kubernetes], sites []
[2015-08-30 10:01:32,153][INFO ][env                      ] [Hammerhead] using [1] data paths, mounts [[/data (/dev/sda9)]], net usable_space [14.4gb], net total_space [15.5gb], types [ext4]
[2015-08-30 10:01:37,188][INFO ][node                     ] [Hammerhead] initialized
[2015-08-30 10:01:37,189][INFO ][node                     ] [Hammerhead] starting ...
[2015-08-30 10:01:37,499][INFO ][transport                ] [Hammerhead] bound_address {inet[/0:0:0:0:0:0:0:0:9300]}, publish_address {inet[/10.244.48.2:9300]}
[2015-08-30 10:01:37,550][INFO ][discovery                ] [Hammerhead] myesdb/n2-6uu_UT3W5XNrjyqBPiA
[2015-08-30 10:01:43,966][INFO ][cluster.service          ] [Hammerhead] new_master [Hammerhead][n2-6uu_UT3W5XNrjyqBPiA][es-kfymw][inet[/10.244.48.2:9300]]{master=true}, reason: zen-disco-join (elected_as_master)
[2015-08-30 10:01:44,010][INFO ][http                     ] [Hammerhead] bound_address {inet[/0:0:0:0:0:0:0:0:9200]}, publish_address {inet[/10.244.48.2:9200]}
[2015-08-30 10:01:44,011][INFO ][node                     ] [Hammerhead] started
[2015-08-30 10:01:44,042][INFO ][gateway                  ] [Hammerhead] recovered [0] indices into cluster_state
```

So we have a 1-node Elasticsearch cluster ready to handle some work.

## Scale

Scaling is as easy as:

```
kubectl scale --replicas=3 rc es
```

Did it work?

```
$ kubectl get pods
NAME             READY     STATUS    RESTARTS   AGE
es-78e0s         1/1       Running   0          8m
es-kfymw         1/1       Running   0          17m
es-rjmer         1/1       Running   0          8m
kube-dns-p3v1u   3/3       Running   0          30m
```

Let's take a look at logs:

```
$ kubectl logs es-kfymw
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
[2015-08-30 10:01:31,946][INFO ][node                     ] [Hammerhead] version[1.7.1], pid[7], build[b88f43f/2015-07-29T09:54:16Z]
[2015-08-30 10:01:31,946][INFO ][node                     ] [Hammerhead] initializing ...
[2015-08-30 10:01:32,110][INFO ][plugins                  ] [Hammerhead] loaded [cloud-kubernetes], sites []
[2015-08-30 10:01:32,153][INFO ][env                      ] [Hammerhead] using [1] data paths, mounts [[/data (/dev/sda9)]], net usable_space [14.4gb], net total_space [15.5gb], types [ext4]
[2015-08-30 10:01:37,188][INFO ][node                     ] [Hammerhead] initialized
[2015-08-30 10:01:37,189][INFO ][node                     ] [Hammerhead] starting ...
[2015-08-30 10:01:37,499][INFO ][transport                ] [Hammerhead] bound_address {inet[/0:0:0:0:0:0:0:0:9300]}, publish_address {inet[/10.244.48.2:9300]}
[2015-08-30 10:01:37,550][INFO ][discovery                ] [Hammerhead] myesdb/n2-6uu_UT3W5XNrjyqBPiA
[2015-08-30 10:01:43,966][INFO ][cluster.service          ] [Hammerhead] new_master [Hammerhead][n2-6uu_UT3W5XNrjyqBPiA][es-kfymw][inet[/10.244.48.2:9300]]{master=true}, reason: zen-disco-join (elected_as_master)
[2015-08-30 10:01:44,010][INFO ][http                     ] [Hammerhead] bound_address {inet[/0:0:0:0:0:0:0:0:9200]}, publish_address {inet[/10.244.48.2:9200]}
[2015-08-30 10:01:44,011][INFO ][node                     ] [Hammerhead] started
[2015-08-30 10:01:44,042][INFO ][gateway                  ] [Hammerhead] recovered [0] indices into cluster_state
[2015-08-30 10:08:02,517][INFO ][cluster.service          ] [Hammerhead] added {[Tenpin][2gv5MiwhRiOSsrTOF3DhuA][es-78e0s][inet[/10.244.54.4:9300]]{master=true},}, reason: zen-disco-receive(join from node[[Tenpin][2gv5MiwhRiOSsrTOF3DhuA][es-78e0s][inet[/10.244.54.4:9300]]{master=true}])
[2015-08-30 10:10:10,645][INFO ][cluster.service          ] [Hammerhead] added {[Evilhawk][ziTq2PzYRJys43rNL2tbyg][es-rjmer][inet[/10.244.33.3:9300]]{master=true},}, reason: zen-disco-receive(join from node[[Evilhawk][ziTq2PzYRJys43rNL2tbyg][es-rjmer][inet[/10.244.33.3:9300]]{master=true}])
```

So we have a 3-node Elasticsearch cluster ready to handle more work.

## Access the service

*Don't forget* that services in Kubernetes are only accessible from containers in the cluster. For different behavior you should [configure the creation of an external load-balancer](http://kubernetes.io/v1.0/docs/user-guide/services.html#type-loadbalancer). While it's supported within this example service descriptor, its usage is out of scope of this document, for now.

```
$ kubectl get service elasticsearch
NAME            LABELS                    SELECTOR                  IP(S)           PORT(S)
elasticsearch   component=elasticsearch   component=elasticsearch   10.100.108.94   9200/TCP
                                                                                    9300/TCP
```

From any host on your cluster (that's running `kube-proxy`), run:

```
$ curl 10.100.108.94:9200
```

You should see something similar to the following:


```json
{
  "status" : 200,
  "name" : "Hammerhead",
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
curl 10.100.108.94:9200/_cluster/health?pretty
```

You should see something similar to the following:

```json
{
  "cluster_name" : "myesdb",
  "status" : "green",
  "timed_out" : false,
  "number_of_nodes" : 3,
  "number_of_data_nodes" : 3,
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
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/elasticsearch/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
