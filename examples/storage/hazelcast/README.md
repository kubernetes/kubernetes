## Cloud Native Deployments of Hazelcast using Kubernetes

The following document describes the development of a _cloud native_ [Hazelcast](http://hazelcast.org/) deployment on Kubernetes.  When we say _cloud native_ we mean an application which understands that it is running within a cluster manager, and uses this cluster management infrastructure to help implement the application. In particular, in this instance, a custom Hazelcast ```bootstrapper``` is used to enable Hazelcast to dynamically discover Hazelcast nodes that have already joined the cluster.

Any topology changes are communicated and handled by Hazelcast nodes themselves.

This document also attempts to describe the core components of Kubernetes: _Pods_, _Services_, and _Deployments_.

### Prerequisites

This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the `kubectl` command line tool somewhere in your path.  Please see the [getting started](../../../docs/getting-started-guides/) for installation instructions for your platform.

### A note for the impatient

This is a somewhat long tutorial.  If you want to jump straight to the "do it now" commands, please see the [tl; dr](#tl-dr) at the end.

### Sources

Source is freely available at:
* Hazelcast Discovery - https://github.com/pires/hazelcast-kubernetes-bootstrapper
* Dockerfile - https://github.com/pires/hazelcast-kubernetes
* Docker Trusted Build - https://quay.io/repository/pires/hazelcast-kubernetes

### Simple Single Pod Hazelcast Node

In Kubernetes, the atomic unit of an application is a [_Pod_](../../../docs/user-guide/pods.md).  A Pod is one or more containers that _must_ be scheduled onto the same host.  All containers in a pod share a network namespace, and may optionally share mounted volumes.

In this case, we shall not run a single Hazelcast pod, because the discovery mechanism now relies on a service definition.


### Adding a Hazelcast Service

In Kubernetes a _[Service](../../../docs/user-guide/services.md)_ describes a set of Pods that perform the same task. For example, the set of nodes in a Hazelcast cluster. An important use for a Service is to create a load balancer which distributes traffic across members of the set.  But a _Service_ can also be used as a standing query which makes a dynamically changing set of Pods available via the Kubernetes API. This is actually how our discovery mechanism works, by relying on the service to discover other Hazelcast pods.

Here is the service description:

<!-- BEGIN MUNGE: EXAMPLE hazelcast-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    name: hazelcast
  name: hazelcast
spec: 
  ports:
    - port: 5701
  selector:
    name: hazelcast
```

[Download example](hazelcast-service.yaml?raw=true)
<!-- END MUNGE: EXAMPLE hazelcast-service.yaml -->

The important thing to note here is the `selector`. It is a query over labels, that identifies the set of _Pods_ contained by the _Service_.  In this case the selector is `name: hazelcast`.  If you look at the Replication Controller specification below, you'll see that the pod has the corresponding label, so it will be selected for membership in this Service.

Create this service as follows:

```sh
$ kubectl create -f examples/storage/hazelcast/hazelcast-service.yaml
```

### Adding replicated nodes

The real power of Kubernetes and Hazelcast lies in easily building a replicated, resizable Hazelcast cluster.

In Kubernetes a _[_Deployment_](../../../docs/user-guide/deployments.md)_ is responsible for replicating sets of identical pods. Like a _Service_ it has a selector query which identifies the members of its set.  Unlike a _Service_ it also has a desired number of replicas, and it will create or delete _Pods_ to ensure that the number of _Pods_ matches up with its desired state.

Deployments will "adopt" existing pods that match their selector query, so let's create a Deployment with a single replica to adopt our existing Hazelcast Pod.

<!-- BEGIN MUNGE: EXAMPLE hazelcast-controller.yaml -->

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata: 
  name: hazelcast
  labels: 
    name: hazelcast
spec: 
  template: 
    metadata: 
      labels: 
        name: hazelcast
    spec: 
      containers: 
      - name: hazelcast
        image: quay.io/pires/hazelcast-kubernetes:0.8.0
        imagePullPolicy: Always
        env:
        - name: "DNS_DOMAIN"
          value: "cluster.local"
        ports: 
        - name: hazelcast
          containerPort: 5701
```

[Download example](hazelcast-deployment.yaml?raw=true)
<!-- END MUNGE: EXAMPLE hazelcast-controller.yaml -->

You may note that we tell Kubernetes that the container exposes the `hazelcast` port.

The bulk of the replication controller config is actually identical to the Hazelcast pod declaration above, it simply gives the controller a recipe to use when creating new pods.  The other parts are the `selector` which contains the controller's selector query, and the `replicas` parameter which specifies the desired number of replicas, in this case 1.

Last but not least, we set `DNS_DOMAIN` environment variable according to your Kubernetes clusters DNS configuration.

Create this controller:

```sh
$ kubectl create -f examples/storage/hazelcast/hazelcast-deployment.yaml
```

After the controller provisions successfully the pod, you can query the service endpoints:
```sh
$ kubectl get endpoints hazelcast -o yaml
apiVersion: v1
kind: Endpoints
metadata:
  creationTimestamp: 2016-12-16T08:57:27Z
  labels:
    name: hazelcast
  name: hazelcast
  namespace: default
  resourceVersion: "11360"
  selfLink: /api/v1/namespaces/default/endpoints/hazelcast
  uid: 46447198-70eb-11e6-940c-0800278ab84d
subsets:
- addresses:
  - ip: 10.244.37.2
    targetRef:
      kind: Pod
      name: hazelcast-1790698550-3heau
      namespace: default
      resourceVersion: "11359"
      uid: c9c3febd-70eb-11e6-940c-0800278ab84d
  ports:
  - port: 5701
    protocol: TCP
```

You can see that the _Service_ has found the pod created by the replication controller.

Now it gets even more interesting. Let's scale our cluster to 2 pods:
```sh
$ kubectl scale deployment hazelcast --replicas 2
```

Now if you list the pods in your cluster, you should see two hazelcast pods:

```sh
$ kubectl get deployment,pods
NAME               DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
deploy/hazelcast   2         2         2            2           1m

NAME                            READY     STATUS    RESTARTS   AGE
po/hazelcast-3980717115-k1xsk   1/1       Running   0          1m
po/hazelcast-3980717115-pbhbq   1/1       Running   0          22s
```

To prove that this all works, you can use the `log` command to examine the logs of one pod, for example:

```sh
kubectl logs -f hazelcast-39807171
15-k1xsk
2017-01-30 12:42:50.774  INFO 6 --- [           main] com.github.pires.hazelcast.Application   : Starting Application on hazelcast-3980717115-k1xsk with PID 6 (/bootstrapper.jar started by root in /)
2017-01-30 12:42:50.781  INFO 6 --- [           main] com.github.pires.hazelcast.Application   : No active profile set, falling back to default profiles: default
2017-01-30 12:42:50.852  INFO 6 --- [           main] s.c.a.AnnotationConfigApplicationContext : Refreshing org.springframework.context.annotation.AnnotationConfigApplicationContext@14514713: startup date [Mon Jan 30 12:42:50 GMT 2017]; root of context hierarchy
2017-01-30 12:42:52.304  INFO 6 --- [           main] o.s.j.e.a.AnnotationMBeanExporter        : Registering beans for JMX exposure on startup
2017-01-30 12:42:52.323  INFO 6 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Asking k8s registry at https://kubernetes.default.svc.cluster.local..
2017-01-30 12:42:52.857  INFO 6 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Found 1 pods running Hazelcast.
2017-01-30 12:42:52.990  INFO 6 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.7.5] Interfaces is disabled, trying to pick one address from TCP-IP config addresses: [10.244.9.2]
2017-01-30 12:42:52.990  INFO 6 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.7.5] Prefer IPv4 stack is true.
2017-01-30 12:42:53.002  INFO 6 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.7.5] Picked [10.244.9.2]:5701, using socket ServerSocket[addr=/0:0:0:0:0:0:0:0,localport=5701], bind any local is true
2017-01-30 12:42:53.032  INFO 6 --- [           main] com.hazelcast.system                     : [10.244.9.2]:5701 [someGroup] [3.7.5] Hazelcast 3.7.5 (20170124 - 111f332) starting at [10.244.9.2]:5701
2017-01-30 12:42:53.032  INFO 6 --- [           main] com.hazelcast.system                     : [10.244.9.2]:5701 [someGroup] [3.7.5] Copyright (c) 2008-2016, Hazelcast, Inc. All Rights Reserved.
2017-01-30 12:42:53.032  INFO 6 --- [           main] com.hazelcast.system                     : [10.244.9.2]:5701 [someGroup] [3.7.5] Configured Hazelcast Serialization version : 1
2017-01-30 12:42:53.343  INFO 6 --- [           main] c.h.s.i.o.impl.BackpressureRegulator     : [10.244.9.2]:5701 [someGroup] [3.7.5] Backpressure is disabled
2017-01-30 12:42:54.273  INFO 6 --- [           main] com.hazelcast.instance.Node              : [10.244.9.2]:5701 [someGroup] [3.7.5] Creating TcpIpJoiner
2017-01-30 12:42:54.507  INFO 6 --- [           main] c.h.s.i.o.impl.OperationExecutorImpl     : [10.244.9.2]:5701 [someGroup] [3.7.5] Starting 2 partition threads
2017-01-30 12:42:54.508  INFO 6 --- [           main] c.h.s.i.o.impl.OperationExecutorImpl     : [10.244.9.2]:5701 [someGroup] [3.7.5] Starting 3 generic threads (1 dedicated for priority tasks)
2017-01-30 12:42:54.525  INFO 6 --- [           main] com.hazelcast.core.LifecycleService      : [10.244.9.2]:5701 [someGroup] [3.7.5] [10.244.9.2]:5701 is STARTING
2017-01-30 12:42:54.529  INFO 6 --- [           main] c.h.n.t.n.NonBlockingIOThreadingModel    : [10.244.9.2]:5701 [someGroup] [3.7.5] TcpIpConnectionManager configured with Non Blocking IO-threading model: 3 input threads and 3 output threads
2017-01-30 12:42:54.578  INFO 6 --- [           main] com.hazelcast.cluster.impl.TcpIpJoiner   : [10.244.9.2]:5701 [someGroup] [3.7.5]


Members [1] {
	Member [10.244.9.2]:5701 - f9cae801-59da-49d9-b8de-7719abb53844 this
}

2017-01-30 12:42:54.660  INFO 6 --- [           main] com.hazelcast.core.LifecycleService      : [10.244.9.2]:5701 [someGroup] [3.7.5] [10.244.9.2]:5701 is STARTED
2017-01-30 12:42:54.662  INFO 6 --- [           main] com.github.pires.hazelcast.Application   : Started Application in 5.078 seconds (JVM running for 5.771)
2017-01-30 12:44:08.780  INFO 6 --- [thread-Acceptor] c.h.nio.tcp.SocketAcceptorThread         : [10.244.9.2]:5701 [someGroup] [3.7.5] Accepting socket connection from /10.244.93.3:45945
2017-01-30 12:44:08.814  INFO 6 --- [cached.thread-1] c.h.nio.tcp.TcpIpConnectionManager       : [10.244.9.2]:5701 [someGroup] [3.7.5] Established socket connection between /10.244.9.2:5701 and /10.244.93.3:45945
2017-01-30 12:44:15.785  INFO 6 --- [ration.thread-0] c.h.internal.cluster.ClusterService      : [10.244.9.2]:5701 [someGroup] [3.7.5]

Members [2] {
	Member [10.244.9.2]:5701 - f9cae801-59da-49d9-b8de-7719abb53844 this
	Member [10.244.93.3]:5701 - 4e15667b-ce17-40c2-b045-abe3fb25d48b
}
```

Now let's scale our cluster to 4 nodes:
```sh
$ kubectl scale deployment hazelcast --replicas 4
```

Examine the status again by checking a node's logs and you should see the 4 members connected. Something like:
```
(...)

Members [4] {
	Member [10.244.9.2]:5701 - f9cae801-59da-49d9-b8de-7719abb53844 this
	Member [10.244.93.3]:5701 - 4e15667b-ce17-40c2-b045-abe3fb25d48b
	Member [10.244.9.3]:5701 - e0f36fa4-16bf-4009-a034-d4e7a4105003
	Member [10.244.93.4]:5701 - 7ac96b48-aa47-4410-885f-1ad0fc3690f0
}
```

### tl; dr;

For those of you who are impatient, here is the summary of the commands we ran in this tutorial.

```sh
kubectl create -f service.yaml
kubectl create -f deployment.yaml
kubectl scale deployment hazelcast --replicas 2
kubectl scale deployment hazelcast --replicas 4
```

### Hazelcast Discovery Source

See [here](https://github.com/pires/hazelcast-kubernetes-bootstrapper/blob/master/src/main/java/com/github/pires/hazelcast/HazelcastDiscoveryController.java)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/storage/hazelcast/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
