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
[here](http://releases.k8s.io/release-1.0/examples/experimental/horizontal-pod-autoscaling/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Horizontal Pod Autoscaler

Horizontal pod autoscaling is an experimental feature in Kubernetes 1.1.
It allows the number of pods in a replication controller or deployment to scale automatically based on observed CPU or memory usage.
<b>Please note that the current API is tentative and will be subject to change before a stable version is released.</b>

In this document we explain how this feature works by walking you through an example of enabling horizontal pod autoscaling with the php-apache server.

## Prerequisites

This example requires a running Kubernetes cluster in the version 1.1 with experimental API enabled on API server (``--runtime-config=experimental/v1alpha1=true``),
and experimental controllers turned on in controller manager (``--enable-experimental=true``).
This can be simply achieved on GCE by exporting ``KUBE_ENABLE_EXPERIMENTAL_API=true`` before running ```kube-up.sh``` script.

The required version of kubectl is also 1.1.


## Step One: Run & expose php-apache server

To demonstrate horizontal pod autoscaler we will use a custom docker image based on php-apache server.
The image can be found [here](image/).
It defines [index.php](image/index.php) page which performs some CPU intensive computations.

First, we will start a replication controller running the image and expose it as an external service:

```console
$ kubectl create -f ./examples/experimental/horizontal-pod-autoscaling/rc-php-apache.yaml
replicationcontrollers/php-apache

$ kubectl expose rc php-apache --port=80 --type=LoadBalancer
NAME         LABELS           SELECTOR         IP(S)     PORT(S)
php-apache   run=php-apache   run=php-apache             80/TCP
```

Now, we will wait some time and verify that both the replication controller and the service were correctly created and are running. We will also determine the IP address of the service:

```console
$ kubectl get pods
NAME               READY     STATUS    RESTARTS   AGE
php-apache-wa3t1   1/1       Running   0          12m

$ kubectl describe services php-apache | grep "LoadBalancer Ingress"
LoadBalancer Ingress:	146.148.24.244
```

We may now check that php-apache server works correctly by calling ``curl`` with the service's IP:

```console
$ curl http://146.148.24.244
OK!
```

Please notice that when exposing the service we assumed that our cluster runs on a provider which supports load balancers (e.g.: on GCE).
If load balancers are not supported (e.g.: on Vagrant), we can expose php-apache service as ``ClusterIP`` and connect to it using the proxy on the master:

```console
$ kubectl expose rc php-apache --port=80 --type=ClusterIP
NAME         LABELS           SELECTOR         IP(S)     PORT(S)
php-apache   run=php-apache   run=php-apache             80/TCP

$ kubectl cluster-info | grep master
Kubernetes master is running at https://146.148.6.215

$ curl -k -u admin:password https://146.148.6.215/api/v1/proxy/namespaces/default/services/php-apache/
OK!
```


## Step Two: Create horizontal pod autoscaler

Now that the server is running, we will create a horizontal pod autoscaler for it.
To create it, we will use the [hpa-php-apache.yaml](hpa-php-apache.yaml) file, which looks like this:

```yaml
apiVersion: experimental/v1alpha1
kind: HorizontalPodAutoscaler
metadata:
  name: php-apache
  namespace: default
spec:
  maxReplicas: 10
  minReplicas: 1
  scaleRef:
    kind: ReplicationController
    name: php-apache
    namespace: default
  target:
    quantity: 100m
    resource: cpu
```

This defines a horizontal pod autoscaler that maintains between 1 and 10 replicas of the Pods
controlled by the php-apache replication controller you created in the first step of these instructions.
Roughly speaking, the horizontal autoscaler will increase and decrease the number of replicas
(via the replication controller) so as to maintain an average CPU utilization across all Pods of 100 millicores.
See [here](../../../docs/proposals/horizontal-pod-autoscaler.md#autoscaling-algorithm) for more details on the algorithm.

Please be aware that this configuration sets the target CPU consumption to 100 milli-cores, while in [rc-php-apache.yaml](rc-php-apache.yaml) each pod requests 200 milli-cores.
As a general rule, the autoscaler's target should be lower than the request.
Otherwise, overloaded pods may not be able to consume more than the autoscaler's target utilization,
thereby preventing the autoscaler from seeing high enough utilization to trigger it to scale up.

We will create the autoscaler by executing the following command:

```console
$ kubectl create -f ./examples/experimental/horizontal-pod-autoscaling/hpa-php-apache.yaml
horizontalpodautoscaler "php-apache" created
```

We may check the current status of autoscaler by running:

```console
$ kubectl get experimental/hpa
NAME         REFERENCE                                   TARGET     CURRENT   MINPODS   MAXPODS   AGE
php-apache   ReplicationController/default/php-apache/   100m cpu   0 cpu     1         10        4m
```

Please note that the current CPU consumption is 0 as we are not sending any requests to the server
(the ``CURRENT`` column shows the average across all the pods controlled by the corresponding replication controller).

## Step Three: Increase load

Now, we will see how the autoscaler reacts on the increased load of the server.
We will start an infinite loop of queries to our server (please run it in a different terminal):

```console
$ while true; do curl http://146.148.6.244; done
```

We may examine, how CPU load was increased (the results should be visible after about 2 minutes) by executing:

```console
$ kubectl get experimental/hpa
NAME         REFERENCE                                   TARGET     CURRENT    MINPODS   MAXPODS   AGE
php-apache   ReplicationController/default/php-apache/   100m cpu   471m cpu   1         10        10m
```

In the case presented here, it bumped CPU consumption to 471 milli-cores.
As a result, the replication controller was resized to 5 replicas:

```console
$ kubectl get rc
CONTROLLER   CONTAINER(S)   IMAGE(S)                    SELECTOR         REPLICAS   AGE
php-apache   php-apache     gcr.io/dev-jsz/php-apache   run=php-apache   5          26m
```

Now, we may increase the load even more by running yet another infinite loop of queries (in yet another terminal):

```console
$ while true; do curl http://146.148.6.244; done
```

In the case presented here, it increased the number of serving pods to 10:

```console
$ kubectl get experimental/hpa
NAME         REFERENCE                                   TARGET     CURRENT    MINPODS   MAXPODS   AGE
php-apache   ReplicationController/default/php-apache/   100m cpu   133m cpu   1         10        15m

$ kubectl get rc
CONTROLLER   CONTAINER(S)   IMAGE(S)                    SELECTOR         REPLICAS   AGE
php-apache   php-apache     gcr.io/dev-jsz/php-apache   run=php-apache   10         31m
```

## Step Four: Stop load

We will finish our example by stopping the user load.
We will terminate both infinite ``while`` loops sending requests to the server and verify the result state:

```console
$ kubectl get experimental/hpa
NAME         REFERENCE                                   TARGET     CURRENT   MINPODS   MAXPODS   AGE
php-apache   ReplicationController/default/php-apache/   100m cpu   0 cpu     1         10        26m

$ kubectl get rc
CONTROLLER   CONTAINER(S)   IMAGE(S)                    SELECTOR         REPLICAS   AGE
php-apache   php-apache     gcr.io/dev-jsz/php-apache   run=php-apache   1          42m
```

As we see, in the presented case CPU utilization dropped to 0, and the number of replicas dropped to 1.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/experimental/horizontal-pod-autoscaling/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
