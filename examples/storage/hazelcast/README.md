## Cloud Native Deployments of Hazelcast using Kubernetes

The following document describes the development of a _cloud native_ [Hazelcast](http://hazelcast.org/) deployment on Kubernetes.  When we say _cloud native_ we mean an application which understands that it is running within a cluster manager, and uses this cluster management infrastructure to help implement the application. In particular, in this instance, a custom Hazelcast ```bootstrapper``` is used to enable Hazelcast to dynamically discover Hazelcast nodes that have already joined the cluster.

Any topology changes are communicated and handled by Hazelcast nodes themselves.

This document also attempts to describe the core components of Kubernetes: _Pods_, _Services_, and _Deployments_.

### Prerequisites

This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the `kubectl` command line tool somewhere in your path.  Please see the [getting started](https://kubernetes.io/docs/getting-started-guides/) for installation instructions for your platform.

### A note for the impatient

This is a somewhat long tutorial.  If you want to jump straight to the "do it now" commands, please see the [tl; dr](#tl-dr) at the end.

### Sources

Source is freely available at:
* Hazelcast Discovery - https://github.com/pires/hazelcast-kubernetes-bootstrapper
* Dockerfile - https://github.com/pires/hazelcast-kubernetes
* Docker Trusted Build - https://quay.io/repository/pires/hazelcast-kubernetes

### Simple Single Pod Hazelcast Node

In Kubernetes, the atomic unit of an application is a [_Pod_](https://kubernetes.io/docs/user-guide/pods.md).  A Pod is one or more containers that _must_ be scheduled onto the same host.  All containers in a pod share a network namespace, and may optionally share mounted volumes.

In this case, we shall not run a single Hazelcast pod, because the discovery mechanism now relies on a service definition.


### Adding a Hazelcast Service

In Kubernetes a _[Service](https://kubernetes.io/docs/user-guide/services.md)_ describes a set of Pods that perform the same task. For example, the set of nodes in a Hazelcast cluster. An important use for a Service is to create a load balancer which distributes traffic across members of the set.  But a _Service_ can also be used as a standing query which makes a dynamically changing set of Pods available via the Kubernetes API. This is actually how our discovery mechanism works, by relying on the service to discover other Hazelcast pods.

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

In Kubernetes a _[_Deployment_](https://kubernetes.io/docs/user-guide/deployments.md)_ is responsible for replicating sets of identical pods. Like a _Service_ it has a selector query which identifies the members of its set.  Unlike a _Service_ it also has a desired number of replicas, and it will create or delete _Pods_ to ensure that the number of _Pods_ matches up with its desired state.

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
  creationTimestamp: 2017-03-15T09:40:11Z
  labels:
    name: hazelcast
  name: hazelcast
  namespace: default
  resourceVersion: "65060"
  selfLink: /api/v1/namespaces/default/endpoints/hazelcast
  uid: 62645b71-0963-11e7-b39c-080027985ce6
subsets:
- addresses:
  - ip: 172.17.0.2
    nodeName: minikube
    targetRef:
      kind: Pod
      name: hazelcast-4195412960-mgqtk
      namespace: default
      resourceVersion: "65058"
      uid: 7043708f-0963-11e7-b39c-080027985ce6
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
deploy/hazelcast   2         2         2            2           2m

NAME                                           READY     STATUS    RESTARTS   AGE
po/hazelcast-4195412960-0tl3w                  1/1       Running   0          7s
po/hazelcast-4195412960-mgqtk                  1/1       Running   0          2m
```

To prove that this all works, you can use the `log` command to examine the logs of one pod, for example:

```sh
kubectl logs -f hazelcast-4195412960-0tl3w
2017-03-15 09:42:45.046  INFO 7 --- [           main] com.github.pires.hazelcast.Application   : Starting Application on hazelcast-4195412960-0tl3w with PID 7 (/bootstrapper.jar started by root in /)
2017-03-15 09:42:45.060  INFO 7 --- [           main] com.github.pires.hazelcast.Application   : No active profile set, falling back to default profiles: default
2017-03-15 09:42:45.128  INFO 7 --- [           main] s.c.a.AnnotationConfigApplicationContext : Refreshing org.springframework.context.annotation.AnnotationConfigApplicationContext@14514713: startup date [Wed Mar 15 09:42:45 GMT 2017]; root of context hierarchy
2017-03-15 09:42:45.989  INFO 7 --- [           main] o.s.j.e.a.AnnotationMBeanExporter        : Registering beans for JMX exposure on startup
2017-03-15 09:42:46.001  INFO 7 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Asking k8s registry at https://kubernetes.default.svc.cluster.local..
2017-03-15 09:42:46.376  INFO 7 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Found 2 pods running Hazelcast.
2017-03-15 09:42:46.458  INFO 7 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.8] Interfaces is disabled, trying to pick one address from TCP-IP config addresses: [172.17.0.6, 172.17.0.2]
2017-03-15 09:42:46.458  INFO 7 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.8] Prefer IPv4 stack is true.
2017-03-15 09:42:46.464  INFO 7 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.8] Picked [172.17.0.6]:5701, using socket ServerSocket[addr=/0:0:0:0:0:0:0:0,localport=5701], bind any local is true
2017-03-15 09:42:46.484  INFO 7 --- [           main] com.hazelcast.system                     : [172.17.0.6]:5701 [someGroup] [3.8] Hazelcast 3.8 (20170217 - d7998b4) starting at [172.17.0.6]:5701
2017-03-15 09:42:46.484  INFO 7 --- [           main] com.hazelcast.system                     : [172.17.0.6]:5701 [someGroup] [3.8] Copyright (c) 2008-2017, Hazelcast, Inc. All Rights Reserved.
2017-03-15 09:42:46.485  INFO 7 --- [           main] com.hazelcast.system                     : [172.17.0.6]:5701 [someGroup] [3.8] Configured Hazelcast Serialization version : 1
2017-03-15 09:42:46.679  INFO 7 --- [           main] c.h.s.i.o.impl.BackpressureRegulator     : [172.17.0.6]:5701 [someGroup] [3.8] Backpressure is disabled
2017-03-15 09:42:47.069  INFO 7 --- [           main] com.hazelcast.instance.Node              : [172.17.0.6]:5701 [someGroup] [3.8] Creating TcpIpJoiner
2017-03-15 09:42:47.182  INFO 7 --- [           main] c.h.s.i.o.impl.OperationExecutorImpl     : [172.17.0.6]:5701 [someGroup] [3.8] Starting 2 partition threads
2017-03-15 09:42:47.189  INFO 7 --- [           main] c.h.s.i.o.impl.OperationExecutorImpl     : [172.17.0.6]:5701 [someGroup] [3.8] Starting 3 generic threads (1 dedicated for priority tasks)
2017-03-15 09:42:47.197  INFO 7 --- [           main] com.hazelcast.core.LifecycleService      : [172.17.0.6]:5701 [someGroup] [3.8] [172.17.0.6]:5701 is STARTING
2017-03-15 09:42:47.253  INFO 7 --- [cached.thread-3] c.hazelcast.nio.tcp.InitConnectionTask   : [172.17.0.6]:5701 [someGroup] [3.8] Connecting to /172.17.0.2:5701, timeout: 0, bind-any: true
2017-03-15 09:42:47.262  INFO 7 --- [cached.thread-3] c.h.nio.tcp.TcpIpConnectionManager       : [172.17.0.6]:5701 [someGroup] [3.8] Established socket connection between /172.17.0.6:58073 and /172.17.0.2:5701
2017-03-15 09:42:54.260  INFO 7 --- [ration.thread-0] com.hazelcast.system                     : [172.17.0.6]:5701 [someGroup] [3.8] Cluster version set to 3.8
2017-03-15 09:42:54.262  INFO 7 --- [ration.thread-0] c.h.internal.cluster.ClusterService      : [172.17.0.6]:5701 [someGroup] [3.8] 

Members [2] {
	Member [172.17.0.2]:5701 - 170f6924-7888-442a-9875-ad4d25659a8a
	Member [172.17.0.6]:5701 - b1b82bfa-86c2-4931-af57-325c10c03b3b this
}

2017-03-15 09:42:56.285  INFO 7 --- [           main] com.hazelcast.core.LifecycleService      : [172.17.0.6]:5701 [someGroup] [3.8] [172.17.0.6]:5701 is STARTED
2017-03-15 09:42:56.287  INFO 7 --- [           main] com.github.pires.hazelcast.Application   : Started Application in 11.831 seconds (JVM running for 12.219)
```

Now let's scale our cluster to 4 nodes:
```sh
$ kubectl scale deployment hazelcast --replicas 4
```

Examine the status again by checking a node's logs and you should see the 4 members connected. Something like:
```
(...)

Members [4] {
	Member [172.17.0.2]:5701 - 170f6924-7888-442a-9875-ad4d25659a8a
	Member [172.17.0.6]:5701 - b1b82bfa-86c2-4931-af57-325c10c03b3b this
	Member [172.17.0.9]:5701 - 0c7530d3-1b5a-4f40-bd59-7187e43c1110
	Member [172.17.0.10]:5701 - ad5c3000-7fd0-4ce7-8194-e9b1c2ed6dda
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
