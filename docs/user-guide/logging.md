<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Logging

## Logging by Kubernetes Components

Kubernetes components, such as kubelet and apiserver, use the [glog](https://godoc.org/github.com/golang/glog) logging library.  Developer conventions for logging severity are described in [docs/devel/logging.md](../devel/logging.md).

## Examining the logs of running containers

The logs of a running container may be fetched using the command `kubectl logs`. For example, given
this pod specification [counter-pod.yaml](../../examples/blog-logging/counter-pod.yaml), which has a container which writes out some text to standard
output every second. (You can find different pod specifications [here](logging-demo/).)

<!-- BEGIN MUNGE: EXAMPLE ../../examples/blog-logging/counter-pod.yaml -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: counter
spec:
  containers:
  - name: count
    image: ubuntu:14.04
    args: [bash, -c, 
           'for ((i = 0; ; i++)); do echo "$i: $(date)"; sleep 1; done']
```

[Download example](../../examples/blog-logging/counter-pod.yaml?raw=true)
<!-- END MUNGE: EXAMPLE ../../examples/blog-logging/counter-pod.yaml -->

we can run the pod:

```console
$ kubectl create -f ./counter-pod.yaml
pods/counter
```

and then fetch the logs:

```console
$ kubectl logs counter
0: Tue Jun  2 21:37:31 UTC 2015
1: Tue Jun  2 21:37:32 UTC 2015
2: Tue Jun  2 21:37:33 UTC 2015
3: Tue Jun  2 21:37:34 UTC 2015
4: Tue Jun  2 21:37:35 UTC 2015
5: Tue Jun  2 21:37:36 UTC 2015
...
```

If a pod has more than one container then you need to specify which container's log files should
be fetched e.g.

```console
$ kubectl logs kube-dns-v3-7r1l9 etcd
2015/06/23 00:43:10 etcdserver: start to snapshot (applied: 30003, lastsnap: 20002)
2015/06/23 00:43:10 etcdserver: compacted log at index 30003
2015/06/23 00:43:10 etcdserver: saved snapshot at index 30003
2015/06/23 02:05:42 etcdserver: start to snapshot (applied: 40004, lastsnap: 30003)
2015/06/23 02:05:42 etcdserver: compacted log at index 40004
2015/06/23 02:05:42 etcdserver: saved snapshot at index 40004
2015/06/23 03:28:31 etcdserver: start to snapshot (applied: 50005, lastsnap: 40004)
2015/06/23 03:28:31 etcdserver: compacted log at index 50005
2015/06/23 03:28:31 etcdserver: saved snapshot at index 50005
2015/06/23 03:28:56 filePurge: successfully removed file default.etcd/member/wal/0000000000000000-0000000000000000.wal
2015/06/23 04:51:03 etcdserver: start to snapshot (applied: 60006, lastsnap: 50005)
2015/06/23 04:51:03 etcdserver: compacted log at index 60006
2015/06/23 04:51:03 etcdserver: saved snapshot at index 60006
...
```

## Cluster level logging to Google Cloud Logging

The getting started guide [Cluster Level Logging to Google Cloud Logging](../getting-started-guides/logging.md)
explains how container logs are ingested into [Google Cloud Logging](https://cloud.google.com/logging/docs/)
and shows how to query the ingested logs.

## Cluster level logging with Elasticsearch and Kibana

The getting started guide [Cluster Level Logging with Elasticsearch and Kibana](../getting-started-guides/logging-elasticsearch.md)
describes how to ingest cluster level logs into Elasticsearch and view them using Kibana.

## Ingesting Application Log Files

Cluster level logging only collects the standard output and standard error output of the applications
running in containers. The guide [Collecting log files within containers with Fluentd](http://releases.k8s.io/release-1.1/contrib/logging/fluentd-sidecar-gcp/README.md) explains how the log files of applications can also be ingested into Google Cloud logging.

## Known issues

Kubernetes does log rotation for Kubernetes components and docker containers. The command `kubectl logs` currently only read the latest logs, not all historical ones.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/logging.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
