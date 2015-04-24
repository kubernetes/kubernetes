## Cloud Native Deployments of Cassandra using Kubernetes

The following document describes the development of a _cloud native_ [Cassandra](http://cassandra.apache.org/) deployment on Kubernetes.  When we say _cloud native_ we mean an application which understands that it is running within a cluster manager, and uses this cluster management infrastructure to help implement the application.  In particular, in this instance, a custom Cassandra ```SeedProvider``` is used to enable Cassandra to dynamically discover new Cassandra nodes as they join the cluster.

This document also attempts to describe the core components of Kubernetes, _Pods_, _Services_ and _Replication Controllers_.

### Prerequisites
This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs/getting-started-guides) for installation instructions for your platform.

### A note for the impatient
This is a somewhat long tutorial.  If you want to jump straight to the "do it now" commands, please see the [tl; dr](#tl-dr) at the end.

### Simple Single Pod Cassandra Node
In Kubernetes, the atomic unit of an application is a [_Pod_](http://docs.k8s.io/pods.md).  A Pod is one or more containers that _must_ be scheduled onto the same host.  All containers in a pod share a network namespace, and may optionally share mounted volumes.  In this simple case, we define a single container running Cassandra for our pod:

```yaml
id: cassandra
kind: Pod
apiVersion: v1beta1
desiredState:
  manifest:
    version: v1beta1
    id: cassandra
    containers:
      - name: cassandra
        image: kubernetes/cassandra:v2
        command:
          - /run.sh
        cpu: 1000
        ports:
          - name: cql
            containerPort: 9042
          - name: thrift
            containerPort: 9160
        env:
          - key: MAX_HEAP_SIZE
            value: 512M
          - key: HEAP_NEWSIZE
            value: 100M
labels:
  name: cassandra
```

There are a few things to note in this description.  First is that we are running the ```kubernetes/cassandra``` image.  This is a standard Cassandra installation on top of Debian.  However it also adds a custom [```SeedProvider```](https://svn.apache.org/repos/asf/cassandra/trunk/src/java/org/apache/cassandra/locator/SeedProvider.java) to Cassandra.  In Cassandra, a ```SeedProvider``` bootstraps the gossip protocol that Cassandra uses to find other nodes.  The ```KubernetesSeedProvider``` discovers the Kubernetes API Server using the built in Kubernetes discovery service, and then uses the Kubernetes API to find new nodes (more on this later)

You may also note that we are setting some Cassandra parameters (```MAX_HEAP_SIZE``` and ```HEAP_NEWSIZE```).  We also tell Kubernetes that the container exposes both the ```CQL``` and ```Thrift``` API ports.  Finally, we tell the cluster manager that we need 1000 milli-cpus (1 core).

Given this configuration, we can create the pod as follows

```sh
$ kubectl create -f cassandra.yaml
```

After a few moments, you should be able to see the pod running:

```sh
$ kubectl get pods cassandra
POD                 CONTAINER(S)        IMAGE(S)               HOST                                                          LABELS              STATUS
cassandra           cassandra           kubernetes/cassandra   kubernetes-minion-1/1.2.3.4   name=cassandra      Running
```


### Adding a Cassandra Service
In Kubernetes a _Service_ describes a set of Pods that perform the same task.  For example, the set of nodes in a Cassandra cluster, or even the single node we created above.  An important use for a Service is to create a load balancer which distributes traffic across members of the set.  But a _Service_ can also be used as a standing query which makes a dynamically changing set of Pods (or the single Pod we've already created) available via the Kubernetes API.  This is the way that we use initially use Services with Cassandra.

Here is the service description:
```yaml
id: cassandra
kind: Service
apiVersion: v1beta1
port: 9042
containerPort: 9042
selector:
  name: cassandra
```

The important thing to note here is the ```selector```. It is a query over labels, that identifies the set of _Pods_ contained by the _Service_.  In this case the selector is ```name=cassandra```.  If you look back at the Pod specification above, you'll see that the pod has the corresponding label, so it will be selected for membership in this Service.

Create this service as follows:
```sh
$ kubectl create -f cassandra-service.yaml
```

Once the service is created, you can query it's endpoints:
```sh
$ kubectl get endpoints cassandra -o yaml
apiVersion: v1beta1
creationTimestamp: 2015-01-05T05:51:50Z
endpoints:
- 10.244.1.10:9042
id: cassandra
kind: Endpoints
namespace: default
resourceVersion: 69130
selfLink: /api/v1beta1/endpoints/cassandra?namespace=default
uid: f1937b47-949e-11e4-8a8b-42010af0e23e
```

You can see that the _Service_ has found the pod we created in step one.

### Adding replicated nodes
Of course, a single node cluster isn't particularly interesting.  The real power of Kubernetes and Cassandra lies in easily building a replicated, resizable Cassandra cluster.

In Kubernetes a _Replication Controller_ is responsible for replicating sets of identical pods.  Like a _Service_ it has a selector query which identifies the members of it's set.  Unlike a _Service_ it also has a desired number of replicas, and it will create or delete _Pods_ to ensure that the number of _Pods_ matches up with it's desired state.

Replication Controllers will "adopt" existing pods that match their selector query, so let's create a Replication Controller with a single replica to adopt our existing Cassandra Pod.

```yaml
id: cassandra
kind: ReplicationController
apiVersion: v1beta1
desiredState:
  replicas: 1
  replicaSelector:
    name: cassandra
  # This is identical to the pod config above
  podTemplate:
    desiredState:
      manifest:
        version: v1beta1
        id: cassandra
        containers:
          - name: cassandra
            image: kubernetes/cassandra:v2
            command:
              - /run.sh
            cpu: 1000
            ports:
              - name: cql
                containerPort: 9042
              - name: thrift
                containerPort: 9160
            env:
              - key: MAX_HEAP_SIZE
                value: 512M
              - key: HEAP_NEWSIZE
                value: 100M
    labels:
      name: cassandra
```

The bulk of the replication controller config is actually identical to the Cassandra pod declaration above, it simply gives the controller a recipe to use when creating new pods.  The other parts are the ```replicaSelector``` which contains the controller's selector query, and the ```replicas``` parameter which specifies the desired number of replicas, in this case 1.

Create this controller:

```sh
$ kubectl create -f cassandra-controller.yaml
```

Now this is actually not that interesting, since we haven't actually done anything new.  Now it will get interesting.

Let's resize our cluster to 2:
```sh
$ kubectl resize rc cassandra --replicas=2
```

Now if you list the pods in your cluster, you should see two cassandra pods:

```sh
$ kubectl get pods
POD                                    CONTAINER(S)        IMAGE(S)                       HOST                                                          LABELS              STATUS
cassandra                              cassandra           kubernetes/cassandra           kubernetes-minion-1.c.my-cloud-code.internal/1.2.3.4   name=cassandra      Running
16b2beab-94a1-11e4-8a8b-42010af0e23e   cassandra           kubernetes/cassandra           kubernetes-minion-3.c.my-cloud-code.internal/2.3.4.5                 name=cassandra      Running
```

Notice that one of the pods has the human readable name ```cassandra``` that you specified in your config before, and one has a random string, since it was named by the replication controller.

To prove that this all works, you can use the ```nodetool``` command to examine the status of the cluster, for example:

```sh
$ ssh 1.2.3.4
$ docker exec <cassandra-container-id> nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address      Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.3.29  72.07 KB   256     100.0%            f736f0b5-bd1f-46f1-9b9d-7e8f22f37c9e  rack1
UN  10.244.1.10  41.14 KB   256     100.0%            42617acd-b16e-4ee3-9486-68a6743657b1  rack1
```

Now let's resize our cluster to 4 nodes:
```sh
$ kubectl resize rc cassandra --replicas=4
```

Examining the status again:
```sh
$ docker exec <cassandra-container-id> nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address      Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.3.29  72.07 KB   256     49.5%             f736f0b5-bd1f-46f1-9b9d-7e8f22f37c9e  rack1
UN  10.244.2.14  61.62 KB   256     52.6%             3e9981a6-6919-42c4-b2b8-af50f23a68f2  rack1
UN  10.244.1.10  41.14 KB   256     49.5%             42617acd-b16e-4ee3-9486-68a6743657b1  rack1
UN  10.244.4.8   63.83 KB   256     48.3%             eeb73967-d1e6-43c1-bb54-512f8117d372  rack1
```

### tl; dr;
For those of you who are impatient, here is the summary of the commands we ran in this tutorial.

```sh
# create a single cassandra node
kubectl create -f cassandra.yaml

# create a service to track all cassandra nodes
kubectl create -f cassandra-service.yaml

# create a replication controller to replicate cassandra nodes
kubectl create -f cassandra-controller.yaml

# scale up to 2 nodes
kubectl resize rc cassandra --replicas=2

# validate the cluster
docker exec <container-id> nodetool status

# scale up to 4 nodes
kubectl resize rc cassandra --replicas=4
```

### Seed Provider Source
```java
package io.k8s.cassandra;

import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.net.URL;
import java.net.URLConnection;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.codehaus.jackson.JsonNode;
import org.codehaus.jackson.annotate.JsonIgnoreProperties;
import org.codehaus.jackson.map.ObjectMapper;
import org.apache.cassandra.locator.SeedProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KubernetesSeedProvider implements SeedProvider {
    @JsonIgnoreProperties(ignoreUnknown = true)
    static class Endpoints {
        public String[] endpoints;
    }
    
    private static String getEnvOrDefault(String var, String def) {
        String val = System.getenv(var);
        if (val == null) {
	    val = def;
        }
        return val;
    }

    private static final Logger logger = LoggerFactory.getLogger(KubernetesSeedProvider.class);

    private List defaultSeeds;
   
    public KubernetesSeedProvider(Map<String, String> params) {
        // Taken from SimpleSeedProvider.java
        // These are used as a fallback, if we get nothing from k8s.
        String[] hosts = params.get("seeds").split(",", -1);
        defaultSeeds = new ArrayList<InetAddress>(hosts.length);
        for (String host : hosts)
	    {
		try {
		    defaultSeeds.add(InetAddress.getByName(host.trim()));
		}
		catch (UnknownHostException ex)
		    {
			// not fatal... DD will bark if there end up being zero seeds.
			logger.warn("Seed provider couldn't lookup host " + host);
		    }
	    }
    } 

    public List<InetAddress> getSeeds() {
        List<InetAddress> list = new ArrayList<InetAddress>();
        String protocol = getEnvOrDefault("KUBERNETES_API_PROTOCOL", "http");
        String hostName = getEnvOrDefault("KUBERNETES_RO_SERVICE_HOST", "localhost");
        String hostPort = getEnvOrDefault("KUBERNETES_RO_SERVICE_PORT", "8080");

        String host = protocol + "://" + hostName + ":" + hostPort;
        String serviceName = getEnvOrDefault("CASSANDRA_SERVICE", "cassandra");
        String path = "/api/v1beta1/endpoints/";
        try {
	    URL url = new URL(host + path + serviceName);
	    ObjectMapper mapper = new ObjectMapper();
	    Endpoints endpoints = mapper.readValue(url, Endpoints.class);
	    if (endpoints != null) {
            // Here is a problem point, endpoints.endpoints can be null in first node cases.
            if (endpoints.endpoints != null){
		for (String endpoint : endpoints.endpoints) {
		    String[] parts = endpoint.split(":");
		    list.add(InetAddress.getByName(parts[0]));
		}
            }
	    }
        } catch (IOException ex) {
	    logger.warn("Request to kubernetes apiserver failed"); 
        }
        if (list.size() == 0) {
	    // If we got nothing, we might be the first instance, in that case
	    // fall back on the seeds that were passed in cassandra.yaml.
	    return defaultSeeds;
        }
        return list;
    }

    // Simple main to test the implementation
    public static void main(String[] args) {
        SeedProvider provider = new KubernetesSeedProvider(new HashMap<String, String>());
        System.out.println(provider.getSeeds());
    }
}
```
