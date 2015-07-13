<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<h1>*** PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes User Guide: Managing Applications: Connecting applications

Now that you have a continuously running, replicated application you can expose it on a network. Before discussing the Kubernetes approach to networking, it is worthwhile to contrast it with the "normal" way networking works with Docker.

By default, Docker uses host-private networking, so containers can talk to other containers only if they are on the same machine. In order for Docker containers to communicate across nodes, they must be allocated ports on the machine's own IP address, which are then forwarded or proxied to the containers. This obviously means that containers must either coordinate which ports they use very carefully or else be allocated ports dynamically.

Coordinating ports across multiple developers is very difficult to do at scale and exposes users to cluster-level issues outside of their control. Kubernetes assumes that pods can communicate with other pods, regardless of which host they land on. We give every pod its own cluster-private-IP address so you do not need to explicitly create links between pods or mapping container ports to host ports. This means that containers within a Pod can all reach each other’s ports on localhost, and all pods in a cluster can see each other without NAT. The rest of this document will elaborate on how you can run reliable services on such a networking model.

## Exposing nginx pods to the cluster

We did this in a previous example, but lets do it once again and focus on the networking perspective. Create an nginx pod, and note that it has a container port specification:
```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: my-nginx
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
```

This makes it accessible from any node in your cluster. Check the nodes the pod is running on:
```shell
$ kubectl get pods -l app=nginx -o wide
my-nginx-6isf4   1/1       Running   0          2h        e2e-test-beeps-minion-93ly
my-nginx-t26zt   1/1       Running   0          2h        e2e-test-beeps-minion-93ly
```

Check your pods ips:
```shell
$ kubectl get pods -l app=nginx -o json | grep podIP
                "podIP": "10.245.0.15",
                "podIP": "10.245.0.14",
```
You should be able to ssh into any node in your cluster and curl both ips. Note that the containers are *not* using port 80 on the node, nor are there any special NAT rules to route traffic to the pod. This means you can run multiple nginx pods on the same node all using the same containerPort and access them from any other pod or node in your cluster using ip. Like Docker, ports can still be published to the host node's interface(s), but the need for this is radically diminished because of the networking model.

You can read more about [how we achieve this](../networking.md#how-to-achieve-this) if you’re curious.

## Creating a Service for the pods

So we have pods running nginx in a flat, cluster wide, address space. In theory, you could talk to these pods directly, but what happens when a node dies? The pods die with it, and the replication controller will create new ones, with different ips. This is the problem a Service solves.

A Kubernetes Service is an abstraction which defines a logical set of Pods running somewhere in your cluster, that all provide the same functionality. When created, each Service is assigned a unique IP address (also called clusterIP). This address is tied to the lifespan of the Service, and will not change while the Service is alive. Pods can be configured to talk to the Service, and know that communication to the Service will be automatically load-balanced out to some pod that is a member of the Service ([why not use round robin dns?](../services.md#why-not-use-round-robin-dns)).

You can create a Service for your 2 nginx replicas with the following yaml:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginxsvc
spec:
  ports:
  - port: 80
    protocol: TCP
  selector:
    app: nginx
```
This specification will create a Service which targets TCP port 80 on any Pod with the "app=nginx" label, and expose it on a targetPort (set to the `port` by default) that other pods can use to access the Service. Check your Service:
```shell
$ kubectl get svc
NAME         LABELS     SELECTOR    IP(S)          PORT(S)
nginxsvc     <none>     app=nginx   10.0.116.146   80/TCP
```

As mentioned previously, a Service is backed by a group of pods. These pods are exposed through `endpoints`. The Service's selector will be evaluated continuously and the results will be POSTed to an Endpoints object also named `nginxsvc`. When a pod dies, it is automatically removed from the endpoints, and new pods matching the Service’s selector will automatically get added to the endpoints. Check the endpoints, and note that the ips are the same as the pods created in the first step:
```shell
$ kubectl get ep
NAME         ENDPOINTS
nginxsvc     10.245.0.14:80,10.245.0.15:80
```
You should now be able to curl the nginx Service on `10.0.208.159:80` from any node in your cluster. Note that the Service ip is completely virtual, it never hits the wire, if you’re curious about how this works you can read more about the [service proxy](../services.md#virtual-ips-and-service-proxies).

## Accessing the Service from other pods in the cluster

Kubernetes supports 2 primary modes of finding a Service - environment variables and DNS. The former works out of the box while the latter requires the [kube-dns cluster addon](../../cluster/addons/dns/README.md).

### Environment Variables
When a Pod is run on a Node, the kubelet adds a set of environment variables for each active Service. This introduces an ordering problem. To see why, inspect the environment of your running nginx pods:
```shell
$ kubectl exec my-nginx-6isf4 -- printenv | grep SERVICE
KUBERNETES_SERVICE_HOST=10.0.0.1
KUBERNETES_SERVICE_PORT=443
```
Note there’s no mention of your Service. This is because you created the replicas before the Service. Another disadvantage of doing this is that the scheduler might put both pods on the same machine, which will take your entire Service down if it dies. We can do this the right way by killing the 2 pods and waiting for the replication controller to recreate them. This time around the Service exists *before* the replicas. This will given you scheduler level Service spreading of your pods (provided all your nodes have equal capacity), as well as the right environment variables:
```shell
$ kubectl scale rc my-nginx --replicas=0; kubectl scale rc my-nginx --replicas=2;
$ kubectl get pods -l app=nginx -o wide
NAME             READY   STATUS     RESTARTS   AGE   NODE
my-nginx-5j8ok   1/1     Running   	0         2m    node1
my-nginx-90vaf   1/1     Running   0          2m    node2

$ kubectl exec my-nginx-5j8ok -- printenv | grep SERVICE
KUBERNETES_SERVICE_PORT=443
NGINXSVC_SERVICE_HOST=10.0.116.146
KUBERNETES_SERVICE_HOST=10.0.0.1
NGINXSVC_SERVICE_PORT=80
```

### DNS
Kubernetes offers a DNS cluster addon Service that uses skydns to automatically assign dns names to other Services. You can check if it’s running on your cluster:
```shell
$ kubectl get services kube-dns --namespace=kube-system
NAME       LABELS       SELECTOR             IP(S)       PORT(S)
kube-dns   <none>       k8s-app=kube-dns     10.0.0.10   53/UDP
                                                         53/TCP
```
If it isn’t running, you can [enable it](../../cluster/addons/dns/README.md#how-do-i-configure-it). The rest of this section will assume you have a Service with a long lived ip (nginxsvc), and a dns server that has assigned a name to that ip (the kube-dns cluster addon), so you can talk to the Service from any pod in your cluster using standard methods (e.g. gethostbyname). Let’s create another pod to test this:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  namespace: default
spec:
  containers:
  - image: busybox
    command:
      - sleep
      - "3600"
    imagePullPolicy: IfNotPresent
    name: busybox
  restartPolicy: Always
```
And perform a lookup of the nginx Service
```shell
$ kubectl get pods busybox
NAME      READY     STATUS    RESTARTS   AGE
busybox   1/1       Running   0          18s

$ kubectl exec busybox -- nslookup nginxsvc
Server:    10.0.0.10
Address 1: 10.0.0.10
Name:      nginxsvc
Address 1: 10.0.116.146
```

## Exposing the Service to the internet

For some parts of your applications you may want to expose a Service onto an external IP address. Kubernetes supports two ways of doing this: NodePorts and LoadBalancers.
Lets recreate the nginx Service with a node port:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginxsvc
spec:
  type: NodePort
  ports:
  - port: 80
    protocol: TCP
    targetPort: 80
  selector:
    app: nginx
```

If you have [firewall rules](../services-firewalls.md),  you will have to open the tcp port in the output of the kubectl command before you can access your Service:
```shell
$ kubectl delete svc nginxsvc
$ kubectl create -f nginxsvc.yaml
You have exposed your service on an external port on all nodes in your cluster.
If you want to expose this service to the external internet, you may need to set up
firewall rules for the service port(s) (tcp:31308) to serve traffic.
```
This opens 31308 on *all* nodes in your cluster and redirects it to your Service. You can curl the public ip of your nodes at the given port to access the Service, or add all the ip:port combinations to a load balancer like HAProxy.

Lets now recreate the Service to use a cloud load balancer:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginxsvc
spec:
  type: LoadBalancer
  ports:
  - port: 80
    protocol: TCP
    targetPort: 80
  selector:
    app: nginx
```
You should see a similar message informing you about firewall rules on port 80:
```shell
$ kubectl delete svc nginxsvc
$ kubectl create -f nginxsvc.yaml
An external load-balanced service was created.  On many platforms (e.g. Google Compute Engine),
you will also need to explicitly open a Firewall rule for the service port(s) (tcp:80) to serve traffic.

$ kubectl get service nginxsvc -o json | grep \"ip\"
"ip": "104.197.37.222"
```
Now you have a load balancer that automatically does what you would’ve in the previous step. Note that you cannot directly curl your nodes on port 80, you need to go to the ip of the load balancer.



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/connecting-applications.md?pixel)]()
