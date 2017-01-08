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

In Kubernetes a _[Service](../../../docs/user-guide/services.md)_ describes a set of Pods that perform the same task.  For example, the set of nodes in a Hazelcast cluster.  An important use for a Service is to create a load balancer which distributes traffic across members of the set.  But a _Service_ can also be used as a standing query which makes a dynamically changing set of Pods available via the Kubernetes API.  This is actually how our discovery mechanism works, by relying on the service to discover other Hazelcast pods.

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

In Kubernetes a _[_Deployment_](../../../docs/user-guide/deployments.md)_ is responsible for replicating sets of identical pods. Like a _Service_ it has a selector query which identifies the members of it's set.  Unlike a _Service_ it also has a desired number of replicas, and it will create or delete _Pods_ to ensure that the number of _Pods_ matches up with it's desired state.

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
        image: quay.io/pires/hazelcast-kubernetes:0.7.0
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

You may note that we tell Kubernetes that the container exposes the `hazelcast` port. Finally, we tell the cluster manager that we need 1 cpu core.

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

Now it gets even more interesting.

Let's scale our cluster to 2 pods:

```sh
$ kubectl scale deployment hazelcast --replicas=2
```

Now if you list the pods in your cluster, you should see two hazelcast pods:

```sh
$ kubectl get deployment,pods
NAME                         DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
hazelcast                    2         2         2            2           1m
NAME                         READY     STATUS    RESTARTS     AGE
hazelcast-1790698550-3heau   1/1       Running   0            1m
hazelcast-1790698550-hncjj   1/1       Running   0            48s
```

To prove that this all works, you can use the `log` command to examine the logs of one pod, for example:


```sh
$ kubectl logs hazelcast-1790698550-3heau
2016-09-02 09:01:47.401  INFO 5 --- [           main] com.github.pires.hazelcast.Application   : Starting Application on hazelcast-1790698550-3heau with PID 5 (/bootstrapper.jar started by root in /)
2016-09-02 09:01:47.419  INFO 5 --- [           main] com.github.pires.hazelcast.Application   : No active profile set, falling back to default profiles: default
2016-09-02 09:01:47.605  INFO 5 --- [           main] s.c.a.AnnotationConfigApplicationContext : Refreshing org.springframework.context.annotation.AnnotationConfigApplicationContext@46fbb2c1: startup date [Fri Dec 16 09:01:47 GMT 2016]; root of context hierarchy
2016-09-02 09:01:49.577  INFO 5 --- [           main] o.s.j.e.a.AnnotationMBeanExporter        : Registering beans for JMX exposure on startup
2016-09-02 09:01:49.596  INFO 5 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Asking k8s registry at https://kubernetes.default.svc.cluster.local..
2016-09-02 09:01:50.459  INFO 5 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Found 1 pods running Hazelcast.
2016-09-02 09:01:50.704  INFO 5 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.7.4] Interfaces is disabled, trying to pick one address from TCP-IP config addresses: [10.244.37.2]
2016-09-02 09:01:50.704  INFO 5 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.7.4] Prefer IPv4 stack is true.
2016-09-02 09:01:50.720  INFO 5 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.7.4] Picked [10.244.37.2]:5701, using socket ServerSocket[addr=/0:0:0:0:0:0:0:0,localport=5701], bind any local is true
2016-09-02 09:01:50.772  INFO 5 --- [           main] com.hazelcast.system                     : [10.244.37.2]:5701 [someGroup] [3.7.4] Hazelcast 3.7 (20160817 - 1302600) starting at [10.244.37.2]:5701
2016-09-02 09:01:50.772  INFO 5 --- [           main] com.hazelcast.system                     : [10.244.37.2]:5701 [someGroup] [3.7.4] Copyright (c) 2008-2016, Hazelcast, Inc. All Rights Reserved.
2016-09-02 09:01:50.772  INFO 5 --- [           main] com.hazelcast.system                     : [10.244.37.2]:5701 [someGroup] [3.7.4] Configured Hazelcast Serialization version : 1
2016-09-02 09:01:51.280  INFO 5 --- [           main] c.h.s.i.o.impl.BackpressureRegulator     : [10.244.37.2]:5701 [someGroup] [3.7.4] Backpressure is disabled
2016-09-02 09:01:52.508  INFO 5 --- [           main] com.hazelcast.instance.Node              : [10.244.37.2]:5701 [someGroup] [3.7.4] Creating TcpIpJoiner
2016-09-02 09:01:52.510  INFO 5 --- [           main] com.hazelcast.core.LifecycleService      : [10.244.37.2]:5701 [someGroup] [3.7.4] [10.244.37.2]:5701 is STARTING
2016-09-02 09:01:52.869  INFO 5 --- [           main] c.h.s.i.o.impl.OperationExecutorImpl     : [10.244.37.2]:5701 [someGroup] [3.7.4] Starting 2 partition threads
2016-09-02 09:01:52.874  INFO 5 --- [           main] c.h.s.i.o.impl.OperationExecutorImpl     : [10.244.37.2]:5701 [someGroup] [3.7.4] Starting 3 generic threads (1 dedicated for priority tasks)
2016-09-02 09:01:52.893  INFO 5 --- [           main] c.h.n.t.n.NonBlockingIOThreadingModel    : [10.244.37.2]:5701 [someGroup] [3.7.4] TcpIpConnectionManager configured with Non Blocking IO-threading model: 3 input threads and 3 output threads
2016-09-02 09:01:52.945  INFO 5 --- [           main] com.hazelcast.cluster.impl.TcpIpJoiner   : [10.244.37.2]:5701 [someGroup] [3.7.4]


Members [1] {
       	Member [10.244.37.2]:5701 - a212635b-ffd9-4510-99db-3ef75957dbe8 this
}

2016-09-02 09:01:53.044  INFO 5 --- [           main] com.hazelcast.core.LifecycleService      : [10.244.37.2]:5701 [someGroup] [3.7.4] [10.244.37.2]:5701 is STARTED
2016-09-02 09:01:53.049  INFO 5 --- [           main] com.github.pires.hazelcast.Application   : Started Application in 6.954 seconds (JVM running for 7.91)
2016-09-02 09:03:00.832  INFO 5 --- [thread-Acceptor] c.h.nio.tcp.SocketAcceptorThread         : [10.244.37.2]:5701 [someGroup] [3.7.4] Accepting socket connection from /10.244.86.3:37517
2016-09-02 09:03:00.849  INFO 5 --- [cached.thread-2] c.h.nio.tcp.TcpIpConnectionManager       : [10.244.37.2]:5701 [someGroup] [3.7.4] Established socket connection between /10.244.37.2:5701 and /10.244.86.3:37517
2016-09-02 09:03:07.840  INFO 5 --- [ration.thread-0] c.h.internal.cluster.ClusterService      : [10.244.37.2]:5701 [someGroup] [3.7.4]

Members [2] {
       	Member [10.244.37.2]:5701 - a212635b-ffd9-4510-99db-3ef75957dbe8 this
       	Member [10.244.86.3]:5701 - a605bec1-324a-47ea-90d7-a818a5caa418
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
       	Member [10.244.37.2]:5701 - a212635b-ffd9-4510-99db-3ef75957dbe8 this
       	Member [10.244.86.3]:5701 - a605bec1-324a-47ea-90d7-a818a5caa418
       	Member [10.244.86.4]:5701 - fad6fa83-71d8-40d3-8c94-8962ba34a96e
       	Member [10.244.37.3]:5701 - 4bb7cca2-db64-47ab-b22b-b6388666854e
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
