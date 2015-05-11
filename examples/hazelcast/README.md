## Cloud Native Deployments of Hazelcast using Kubernetes

The following document describes the development of a _cloud native_ [Hazelcast](http://hazelcast.org/) deployment on Kubernetes.  When we say _cloud native_ we mean an application which understands that it is running within a cluster manager, and uses this cluster management infrastructure to help implement the application. In particular, in this instance, a custom Hazelcast ```bootstrapper``` is used to enable Hazelcast to dynamically discover Hazelcast nodes that have already joined the cluster.

Any topology changes are communicated and handled by Hazelcast nodes themselves.

This document also attempts to describe the core components of Kubernetes, _Pods_, _Services_ and _Replication Controllers_.

### Prerequisites
This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the `kubectl` command line tool somewhere in your path.  Please see the [getting started](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs/getting-started-guides) for installation instructions for your platform.

### A note for the impatient
This is a somewhat long tutorial.  If you want to jump straight to the "do it now" commands, please see the [tl; dr](#tl-dr) at the end.

### Sources

Source is freely available at:
* Hazelcast Discovery - https://github.com/pires/hazelcast-kubernetes-bootstrapper
* Dockerfile - https://github.com/pires/hazelcast-kubernetes
* Docker Trusted Build - https://registry.hub.docker.com/u/pires/hazelcast-k8s

### Simple Single Pod Hazelcast Node
In Kubernetes, the atomic unit of an application is a [_Pod_](http://docs.k8s.io/pods.md).  A Pod is one or more containers that _must_ be scheduled onto the same host.  All containers in a pod share a network namespace, and may optionally share mounted volumes. 

In this case, we shall not run a single Hazelcast pod, because the discovery mechanism now relies on a service definition.


### Adding a Hazelcast Service
In Kubernetes a _Service_ describes a set of Pods that perform the same task.  For example, the set of nodes in a Hazelcast cluster.  An important use for a Service is to create a load balancer which distributes traffic across members of the set.  But a _Service_ can also be used as a standing query which makes a dynamically changing set of Pods available via the Kubernetes API.  This is actually how our discovery mechanism works, by relying on the service to discover other Hazelcast pods.

Here is the service description:
```yaml
apiVersion: v1beta3
kind: Service
metadata: 
  labels: 
    name: hazelcast
  name: hazelcast
spec: 
  ports:
    - port: 5701
      targetPort: 5701
  selector: 
    name: hazelcast
```

The important thing to note here is the `selector`. It is a query over labels, that identifies the set of _Pods_ contained by the _Service_.  In this case the selector is `name: hazelcast`.  If you look at the Replication Controller specification below, you'll see that the pod has the corresponding label, so it will be selected for membership in this Service.

Create this service as follows:
```sh
$ kubectl create -f hazelcast-service.yaml
```

### Adding replicated nodes
The real power of Kubernetes and Hazelcast lies in easily building a replicated, resizable Hazelcast cluster.

In Kubernetes a _Replication Controller_ is responsible for replicating sets of identical pods.  Like a _Service_ it has a selector query which identifies the members of it's set.  Unlike a _Service_ it also has a desired number of replicas, and it will create or delete _Pods_ to ensure that the number of _Pods_ matches up with it's desired state.

Replication Controllers will "adopt" existing pods that match their selector query, so let's create a Replication Controller with a single replica to adopt our existing Hazelcast Pod.

```yaml
apiVersion: v1beta3
kind: ReplicationController
metadata: 
  labels: 
    name: hazelcast
  name: hazelcast
spec: 
  replicas: 1
  selector: 
    name: hazelcast
  template: 
    metadata: 
      labels: 
        name: hazelcast
    spec: 
      containers: 
        - resources:
            limits:
              cpu: 1
          image: pires/hazelcast-k8s:0.2
          name: hazelcast
          ports: 
            - containerPort: 5701
              name: hazelcast
```

There are a few things to note in this description.  First is that we are running the `pires/hazelcast-k8s` image, tag `0.2`.  This is a `busybox` installation with JRE 8.  However it also adds a custom [`application`](https://github.com/pires/hazelcast-kubernetes-bootstrapper) that finds any Hazelcast nodes in the cluster and bootstraps an Hazelcast instance accordingle.  The `HazelcastDiscoveryController` discovers the Kubernetes API Server using the built in Kubernetes discovery service, and then uses the Kubernetes API to find new nodes (more on this later).

You may also note that we tell Kubernetes that the container exposes the `hazelcast` port.  Finally, we tell the cluster manager that we need 1 cpu core.

The bulk of the replication controller config is actually identical to the Hazelcast pod declaration above, it simply gives the controller a recipe to use when creating new pods.  The other parts are the ```selector``` which contains the controller's selector query, and the ```replicas``` parameter which specifies the desired number of replicas, in this case 1.

Create this controller:

```sh
$ kubectl create -f hazelcast-controller.yaml
```

After the controller provisions successfully the pod, you can query the service endpoints:
```sh
$ kubectl get endpoints hazelcast -o yaml
apiVersion: v1beta3
kind: Endpoints
metadata:
  creationTimestamp: 2015-05-04T17:43:40Z
  labels:
    name: hazelcast
  name: hazelcast
  namespace: default
  resourceVersion: "120480"
  selfLink: /api/v1beta3/namespaces/default/endpoints/hazelcast
  uid: 19a22aa9-f285-11e4-b38f-42010af0bbf9
subsets:
- addresses:
  - IP: 10.245.2.68
    targetRef:
      kind: Pod
      name: hazelcast
      namespace: default
      resourceVersion: "120479"
      uid: d7238173-f283-11e4-b38f-42010af0bbf9
  ports:
  - port: 5701
    protocol: TCP
```

You can see that the _Service_ has found the pod created by the replication controller.

Now it gets even more interesting.

Let's resize our cluster to 2 pods:
```sh
$ kubectl resize rc hazelcast --replicas=2
```

Now if you list the pods in your cluster, you should see two hazelcast pods:

```sh
$ kubectl get pods
POD                 IP            CONTAINER(S)   IMAGE(S)              HOST                                 LABELS           STATUS    CREATED      MESSAGE
hazelcast-pkyzd     10.244.90.3                                        e2e-test-minion-vj7k/104.197.8.214   name=hazelcast   Running   14 seconds
                                  hazelcast      pires/hazelcast-k8s:0.2                                                         Running   2 seconds
hazelcast-ulkws     10.244.66.2                                        e2e-test-minion-2x1f/146.148.62.37   name=hazelcast   Running   7 seconds    
                                  hazelcast      pires/hazelcast-k8s:0.2                                                         Running   6 seconds
```

To prove that this all works, you can use the `log` command to examine the logs of one pod, for example:

```sh
$ kubectl log hazelcast-ulkws hazelcast
2015-05-09 22:06:20.016  INFO 5 --- [           main] com.github.pires.hazelcast.Application   : Starting Application v0.2-SNAPSHOT on hazelcast-enyli with PID 5 (/bootstrapper.jar started by root in /)
2015-05-09 22:06:20.071  INFO 5 --- [           main] s.c.a.AnnotationConfigApplicationContext : Refreshing org.springframework.context.annotation.AnnotationConfigApplicationContext@5424f110: startup date [Sat May 09 22:06:20 GMT 2015]; root of context hierarchy
2015-05-09 22:06:21.511  INFO 5 --- [           main] o.s.j.e.a.AnnotationMBeanExporter        : Registering beans for JMX exposure on startup
2015-05-09 22:06:21.549  INFO 5 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Asking k8s registry at http://10.100.0.1:80..
2015-05-09 22:06:22.031  INFO 5 --- [           main] c.g.p.h.HazelcastDiscoveryController     : Found 2 pods running Hazelcast.
2015-05-09 22:06:22.176  INFO 5 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.4.2] Interfaces is disabled, trying to pick one address from TCP-IP config addresses: [10.244.90.3, 10.244.66.2]
2015-05-09 22:06:22.177  INFO 5 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.4.2] Prefer IPv4 stack is true.
2015-05-09 22:06:22.189  INFO 5 --- [           main] c.h.instance.DefaultAddressPicker        : [LOCAL] [someGroup] [3.4.2] Picked Address[10.244.66.2]:5701, using socket ServerSocket[addr=/0:0:0:0:0:0:0:0,localport=5701], bind any local is true
2015-05-09 22:06:22.642  INFO 5 --- [           main] com.hazelcast.spi.OperationService       : [10.244.66.2]:5701 [someGroup] [3.4.2] Backpressure is disabled
2015-05-09 22:06:22.647  INFO 5 --- [           main] c.h.spi.impl.BasicOperationScheduler     : [10.244.66.2]:5701 [someGroup] [3.4.2] Starting with 2 generic operation threads and 2 partition operation threads.
2015-05-09 22:06:22.796  INFO 5 --- [           main] com.hazelcast.system                     : [10.244.66.2]:5701 [someGroup] [3.4.2] Hazelcast 3.4.2 (20150326 - f6349a4) starting at Address[10.244.66.2]:5701
2015-05-09 22:06:22.798  INFO 5 --- [           main] com.hazelcast.system                     : [10.244.66.2]:5701 [someGroup] [3.4.2] Copyright (C) 2008-2014 Hazelcast.com
2015-05-09 22:06:22.800  INFO 5 --- [           main] com.hazelcast.instance.Node              : [10.244.66.2]:5701 [someGroup] [3.4.2] Creating TcpIpJoiner
2015-05-09 22:06:22.801  INFO 5 --- [           main] com.hazelcast.core.LifecycleService      : [10.244.66.2]:5701 [someGroup] [3.4.2] Address[10.244.66.2]:5701 is STARTING
2015-05-09 22:06:23.108  INFO 5 --- [cached.thread-2] com.hazelcast.nio.tcp.SocketConnector    : [10.244.66.2]:5701 [someGroup] [3.4.2] Connecting to /10.244.90.3:5701, timeout: 0, bind-any: true
2015-05-09 22:06:23.182  INFO 5 --- [cached.thread-2] c.h.nio.tcp.TcpIpConnectionManager       : [10.244.66.2]:5701 [someGroup] [3.4.2] Established socket connection between /10.244.66.2:48051 and 10.244.90.3/10.244.90.3:5701
2015-05-09 22:06:29.158  INFO 5 --- [ration.thread-1] com.hazelcast.cluster.ClusterService     : [10.244.66.2]:5701 [someGroup] [3.4.2]

Members [2] {
  Member [10.244.90.3]:5701
  Member [10.244.66.2]:5701 this
}

2015-05-09 22:06:31.177  INFO 5 --- [           main] com.hazelcast.core.LifecycleService      : [10.244.66.2]:5701 [someGroup] [3.4.2] Address[10.244.66.2]:5701 is STARTED
```

Now let's resize our cluster to 4 nodes:
```sh
$ kubectl resize rc hazelcast --replicas=4
```

Examine the status again by checking a node’s log and you should see the 4 members connected.

### tl; dr;
For those of you who are impatient, here is the summary of the commands we ran in this tutorial.

```sh
# create a service to track all hazelcast nodes
kubectl create -f hazelcast-service.yaml

# create a replication controller to replicate hazelcast nodes
kubectl create -f hazelcast-controller.yaml

# scale up to 2 nodes
kubectl resize rc hazelcast --replicas=2

# scale up to 4 nodes
kubectl resize rc hazelcast --replicas=4
```

### Hazelcast Discovery Source

```java
package com.github.pires.hazelcast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.hazelcast.config.Config;
import com.hazelcast.config.GroupConfig;
import com.hazelcast.config.JoinConfig;
import com.hazelcast.config.MulticastConfig;
import com.hazelcast.config.NetworkConfig;
import com.hazelcast.config.SSLConfig;
import com.hazelcast.config.TcpIpConfig;
import com.hazelcast.core.Hazelcast;
import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Controller;

/**
 * Read from Kubernetes API all Hazelcast service bound pods, get their IP and connect to them.
 */
@Controller
public class HazelcastDiscoveryController implements CommandLineRunner {

  private static final Logger log = LoggerFactory.getLogger(
      HazelcastDiscoveryController.class);

  @JsonIgnoreProperties(ignoreUnknown = true)
  static class Address {

    public String IP;
  }

  @JsonIgnoreProperties(ignoreUnknown = true)
  static class Subset {

    public List<Address> addresses;
  }

  @JsonIgnoreProperties(ignoreUnknown = true)
  static class Endpoints {

    public List<Subset> subsets;
  }

  private static String getEnvOrDefault(String var, String def) {
    final String val = System.getenv(var);
    return (val == null || val.isEmpty())
        ? def
        : val;
  }

  @Override
  public void run(String... args) {
    final String hostName = getEnvOrDefault("KUBERNETES_RO_SERVICE_HOST",
        "localhost");
    final String hostPort = getEnvOrDefault("KUBERNETES_RO_SERVICE_PORT",
        "8080");
    String serviceName = getEnvOrDefault("HAZELCAST_SERVICE", "hazelcast");
    String path = "/api/v1beta3/namespaces/default/endpoints/";
    final String host = "http://" + hostName + ":" + hostPort;
    log.info("Asking k8s registry at {}..", host);

    final List<String> hazelcastEndpoints = new CopyOnWriteArrayList<>();

    try {
      URL url = new URL(host + path + serviceName);
      ObjectMapper mapper = new ObjectMapper();
      Endpoints endpoints = mapper.readValue(url, Endpoints.class);
      if (endpoints != null) {
        if (endpoints.subsets != null && !endpoints.subsets.isEmpty()) {
          endpoints.subsets.parallelStream().forEach(subset -> {
            subset.addresses.parallelStream().forEach(
                addr -> hazelcastEndpoints.add(addr.IP));
          });
        }
      }
    } catch (IOException ex) {
      log.warn("Request to Kubernetes API failed", ex);
    }

    log.info("Found {} pods running Hazelcast.", hazelcastEndpoints.size());

    runHazelcast(hazelcastEndpoints);
  }

  private void runHazelcast(final List<String> nodes) {
    // configure Hazelcast instance
    final Config cfg = new Config();
    cfg.setInstanceName(UUID.randomUUID().toString());
    // group configuration
    final String HC_GROUP_NAME = getEnvOrDefault("HC_GROUP_NAME", "someGroup");
    final String HC_GROUP_PASSWORD = getEnvOrDefault("HC_GROUP_PASSWORD",
        "someSecret");
    final int HC_PORT = Integer.parseInt(getEnvOrDefault("HC_PORT", "5701"));
    cfg.setGroupConfig(new GroupConfig(HC_GROUP_NAME, HC_GROUP_PASSWORD));
    // network configuration initialization
    final NetworkConfig netCfg = new NetworkConfig();
    netCfg.setPortAutoIncrement(false);
    netCfg.setPort(HC_PORT);
    // multicast
    final MulticastConfig mcCfg = new MulticastConfig();
    mcCfg.setEnabled(false);
    // tcp
    final TcpIpConfig tcpCfg = new TcpIpConfig();
    nodes.parallelStream().forEach(tcpCfg::addMember);
    tcpCfg.setEnabled(true);
    // network join configuration
    final JoinConfig joinCfg = new JoinConfig();
    joinCfg.setMulticastConfig(mcCfg);
    joinCfg.setTcpIpConfig(tcpCfg);
    netCfg.setJoin(joinCfg);
    // ssl
    netCfg.setSSLConfig(new SSLConfig().setEnabled(false));
    // set it all
    cfg.setNetworkConfig(netCfg);
    // run
    Hazelcast.newHazelcastInstance(cfg);
  }

}

```
