## Overview
This example shows two types of pod health checks: HTTP checks and container execution checks.

The [exec-liveness.yaml](./exec-liveness.yaml) demonstrates the container execution check.
```
    livenessProbe:
      exec:
        command:
        - cat
        - /tmp/health
      initialDelaySeconds: 15
      timeoutSeconds: 1
```
Kubelet executes the command cat /tmp/health in the container and reports failure if the command returns a non-zero exit code.

Note that the container removes the /tmp/health file after 10 seconds,
```
echo ok > /tmp/health; sleep 10; rm -rf /tmp/health; sleep 600
```
so when Kubelet executes the health check 15 seconds (defined by initialDelaySeconds) after the container started, the check would fail.


The [http-liveness.yaml](http-liveness.yaml) demonstrates the HTTP check.
```
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 15
      timeoutSeconds: 1
```
The Kubelet sends a HTTP request to the specified path and port to perform the health check. If you take a look at image/server.go, you will see the server starts to respond with an error code 500 after 10 seconds, so the check fails.

This [guide](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/walkthrough/k8s201.md#health-checking) has more information on health checks.

## Get your hands dirty
To show the health check is actually working, first create the pods:
```
# kubectl create -f exec-liveness.yaml
# cluster/kbuectl.sh create -f http-liveness.yaml
```

Check the status of the pods once they are created:
```
# kubectl get pods
POD             IP           CONTAINER(S)   IMAGE(S)                            HOST                                     LABELS          STATUS    CREATED     MESSAGE
liveness-exec   10.244.3.7                                                      kubernetes-minion-f08h/130.211.122.180   test=liveness   Running   3 seconds   
                             liveness       gcr.io/google_containers/busybox                                                             Running   2 seconds   
liveness-http   10.244.0.8                                                      kubernetes-minion-0bks/104.197.10.10     test=liveness   Running   3 seconds   
                             liveness       gcr.io/google_containers/liveness                                                            Running   2 seconds   
```

Check the status half a minute later, you will see the termination messages:
```
# kubectl get pods
POD             IP           CONTAINER(S)   IMAGE(S)                            HOST                                     LABELS          STATUS    CREATED      MESSAGE
liveness-exec   10.244.3.7                                                      kubernetes-minion-f08h/130.211.122.180   test=liveness   Running   34 seconds   
                             liveness       gcr.io/google_containers/busybox                                                             Running   3 seconds    last termination: exit code 137
liveness-http   10.244.0.8                                                      kubernetes-minion-0bks/104.197.10.10     test=liveness   Running   34 seconds   
                             liveness       gcr.io/google_containers/liveness                                                            Running   13 seconds   last termination: exit code 2
```
The termination messages indicate that the liveness probes have failed, and the containers have been killed and recreated.

You can also see the container restart count being incremented by running `kubectl describe`.
```
# kubectl describe pods liveness-exec | grep "Restart Count"
Restart Count:      8
```

You would also see the killing and creating events at the bottom of the *kubectl describe* output:
```
  Thu, 14 May 2015 15:23:25 -0700       Thu, 14 May 2015 15:23:25 -0700 1       {kubelet kubernetes-minion-0uzf}        spec.containers{liveness}               killing      Killing 88c8b717d8b0940d52743c086b43c3fad0d725a36300b9b5f0ad3a1c8cef2d3e
  Thu, 14 May 2015 15:23:25 -0700       Thu, 14 May 2015 15:23:25 -0700 1       {kubelet kubernetes-minion-0uzf}        spec.containers{liveness}               created      Created with docker id b254a9810073f9ee9075bb38ac29a4b063647176ad9eabd9184078ca98a60062
  Thu, 14 May 2015 15:23:25 -0700       Thu, 14 May 2015 15:23:25 -0700 1       {kubelet kubernetes-minion-0uzf}        spec.containers{liveness}               started      Started with docker id b254a9810073f9ee9075bb38ac29a4b063647176ad9eabd9184078ca98a60062
  ...
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/liveness/README.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/examples/liveness/README.md?pixel)]()
