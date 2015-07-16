<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
﻿#Getting into containers: kubectl exec
Developers can use `kubectl exec` to run commands in a container. This guide demonstrates two use cases.

##Using kubectl exec to check the environment variables of a container
Kubernetes exposes [services](services.md#environment-variables) through environment variables. It is convenient to check these environment variables using `kubectl exec`.


We first create a pod and a service,
```
$ kubectl create -f examples/guestbook/redis-master-controller.yaml
$ kubectl create -f examples/guestbook/redis-master-service.yaml
```
wait until the pod is Running and Ready,
```
$ kubectl get pod
NAME                 READY     REASON       RESTARTS   AGE
redis-master-ft9ex   1/1       Running      0          12s
```
then we can check the environment variables of the pod, 
```
$ kubectl exec redis-master-ft9ex env
...
REDIS_MASTER_SERVICE_PORT=6379
REDIS_MASTER_SERVICE_HOST=10.0.0.219
...
```
We can use these environment variables in applications to find the service.


## Using kubectl exec to check the mounted volumes
It is convenient to use `kubectl exec` to check if the volumes are mounted as expected.
We first create a Pod with a volume mounted at /data/redis,
```
kubectl create -f docs/user-guide/walkthrough/pod-redis.yaml
```
wait until the pod is Running and Ready,
```
$ kubectl get pods
NAME      READY     REASON    RESTARTS   AGE
storage   1/1       Running   0          1m
```
we then use `kubectl exec` to verify that the volume is mounted at /data/redis,
```
$ kubectl exec storage ls /data
redis
```

## Using kubectl exec to open a bash terminal in a pod
After all, open a terminal in a pod is the most direct way to introspect the pod. Assuming the pod/storage is still running, run
```
$ kubectl exec -ti storage -- bash
root@storage:/data#
```
This gets you a terminal.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/getting-into-containers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
