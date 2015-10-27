<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Overview

This example shows two types of pod [health checks](../production-pods.md#liveness-and-readiness-probes-aka-health-checks): HTTP checks and container execution checks.

The [exec-liveness.yaml](exec-liveness.yaml) demonstrates the container execution check.

```yaml
    livenessProbe:
      exec:
        command:
        - cat
        - /tmp/health
      initialDelaySeconds: 15
      timeoutSeconds: 1
```

Kubelet executes the command `cat /tmp/health` in the container and reports failure if the command returns a non-zero exit code.

Note that the container removes the `/tmp/health` file after 10 seconds,

```sh
echo ok > /tmp/health; sleep 10; rm -rf /tmp/health; sleep 600
```

so when Kubelet executes the health check 15 seconds (defined by initialDelaySeconds) after the container started, the check would fail.


The [http-liveness.yaml](http-liveness.yaml) demonstrates the HTTP check.

```yaml
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 15
      timeoutSeconds: 1
```

The Kubelet sends an HTTP request to the specified path and port to perform the health check. If you take a look at image/server.go, you will see the server starts to respond with an error code 500 after 10 seconds, so the check fails. The Kubelet sends the probe to the container's ip address by default which could be specified with `host` as part of httpGet probe. If the container listens on `127.0.0.1`, `host` should be specified as `127.0.0.1`. In general, if the container listens on its ip address or on all interfaces (0.0.0.0), there is no need to specify the `host` as part of the httpGet probe.

This [guide](../walkthrough/k8s201.md#health-checking) has more information on health checks.

## Get your hands dirty

To show the health check is actually working, first create the pods:

```console
$ kubectl create -f docs/user-guide/liveness/exec-liveness.yaml
$ kubectl create -f docs/user-guide/liveness/http-liveness.yaml
```

Check the status of the pods once they are created:

```console
$ kubectl get pods
NAME                                           READY     STATUS       RESTARTS   AGE
[...]
liveness-exec                                  1/1       Running      0          13s
liveness-http                                  1/1       Running      0          13s
```

Check the status half a minute later, you will see the container restart count being incremented:

```console
$ kubectl get pods
NAME                                           READY     STATUS       RESTARTS   AGE
[...]
liveness-exec                                  1/1       Running      1          36s
liveness-http                                  1/1       Running      1          36s
```

At the bottom of the *kubectl describe* output there are messages indicating that the liveness probes have failed, and the containers have been killed and recreated.

```console
$ kubectl describe pods liveness-exec
[...]
Sat, 27 Jun 2015 13:43:03 +0200    Sat, 27 Jun 2015 13:44:34 +0200    4    {kubelet kubernetes-minion-6fbi}    spec.containers{liveness}    unhealthy  Liveness probe failed: cat: can't open '/tmp/health': No such file or directory
Sat, 27 Jun 2015 13:44:44 +0200    Sat, 27 Jun 2015 13:44:44 +0200    1    {kubelet kubernetes-minion-6fbi}    spec.containers{liveness}    killing    Killing with docker id 65b52d62c635
Sat, 27 Jun 2015 13:44:44 +0200    Sat, 27 Jun 2015 13:44:44 +0200    1    {kubelet kubernetes-minion-6fbi}    spec.containers{liveness}    created    Created with docker id ed6bb004ee10
Sat, 27 Jun 2015 13:44:44 +0200    Sat, 27 Jun 2015 13:44:44 +0200    1    {kubelet kubernetes-minion-6fbi}    spec.containers{liveness}    started    Started with docker id ed6bb004ee10
```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/liveness/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
