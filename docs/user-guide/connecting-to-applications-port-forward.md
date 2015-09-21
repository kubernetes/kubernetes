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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/connecting-to-applications-port-forward.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Connecting to applications: kubectl port-forward

kubectl port-forward forwards connections to a local port to a port on a pod. Its man page is available [here](kubectl/kubectl_port-forward.md). Compared to [kubectl proxy](accessing-the-cluster.md#using-kubectl-proxy), `kubectl port-forward` is more generic as it can forward TCP traffic while `kubectl proxy` can only forward HTTP traffic. This guide demonstrates how to use `kubectl port-forward` to connect to a Redis database, which may be useful for database debugging.

## Creating a Redis master

```console
$ kubectl create examples/redis/redis-master.yaml
pods/redis-master
```

wait until the Redis master pod is Running and Ready,

```console
$ kubectl get pods
NAME           READY     STATUS    RESTARTS   AGE
redis-master   2/2       Running   0          41s
```


## Connecting to the Redis master[a]

The Redis master is listening on port 6397, to verify this,

```console
$ kubectl get pods redis-master -t='{{(index (index .spec.containers 0).ports 0).containerPort}}{{"\n"}}'
6379
```


then we forward the port 6379 on the local workstation to the port 6379 of pod redis-master,

```console
$ kubectl port-forward redis-master 6379:6379
I0710 14:43:38.274550    3655 portforward.go:225] Forwarding from 127.0.0.1:6379 -> 6379
I0710 14:43:38.274797    3655 portforward.go:225] Forwarding from [::1]:6379 -> 6379
```

To verify the connection is successful, we run a redis-cli on the local workstation,

```console
$ redis-cli
127.0.0.1:6379> ping
PONG
```

Now one can debug the database from the local workstation.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/connecting-to-applications-port-forward.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
