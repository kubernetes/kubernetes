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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/examples/guestbook/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->


## Guestbook Example

This example shows how to build a simple, multi-tier web application using Kubernetes and [Docker](https://www.docker.com/).

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

  - [Guestbook Example](#guestbook-example)
    - [Prerequisites](#prerequisites)
    - [Quick Start](#quick-start)
    - [Step One: Start up the redis master](#step-one-start-up-the-redis-master)
      - [Define a replication controller](#define-a-replication-controller)
      - [Define a service](#define-a-service)
      - [Create a service](#create-a-service)
      - [Finding a service](#finding-a-service)
      - [Create a replication controller](#create-a-replication-controller)
      - [Optional Interlude](#optional-interlude)
    - [Step Two: Start up the redis slave](#step-two-start-up-the-redis-slave)
    - [Step Three: Start up the guestbook frontend](#step-three-start-up-the-guestbook-frontend)
      - [Using 'type: LoadBalancer' for the frontend service (cloud-provider-specific)](#using-type-loadbalancer-for-the-frontend-service-cloud-provider-specific)
    - [Step Four: Cleanup](#step-four-cleanup)
    - [Troubleshooting](#troubleshooting)
    - [Appendix: Accessing the guestbook site externally](#appendix-accessing-the-guestbook-site-externally)
      - [Google Compute Engine External Load Balancer Specifics](#google-compute-engine-external-load-balancer-specifics)

<!-- END MUNGE: GENERATED_TOC -->

The example consists of:

- A web frontend
- A [redis](http://redis.io/) master (for storage), and a replicated set of redis 'slaves'.

The web frontend interacts with the redis master via javascript redis API calls.

**Note**:  If you are running this example on a [Google Container Engine](https://cloud.google.com/container-engine/) installation, see [this Container Engine guestbook walkthrough](https://cloud.google.com/container-engine/docs/tutorials/guestbook) instead. The basic concepts are the same, but the walkthrough is tailored to a Container Engine setup.

### Prerequisites

This example requires a running Kubernetes cluster. See the [Getting Started guides](../../docs/getting-started-guides/) for how to get started. And follow the [Prerequisites](../../docs/user-guide/prereqs.md) to make sure your `kubectl` is ok. As noted above, if you have a Google Container Engine cluster set up, go [here](https://cloud.google.com/container-engine/docs/tutorials/guestbook) instead.

### Quick Start

This section shows a simplest way to get the example work. If you want to know the details, you should skip this and read [the rest of the example](#step-one-start-up-the-redis-master).

Start the guestbook with one command:

```console
$ kubectl create -f examples/guestbook/all-in-one/guestbook-all-in-one.yaml
service "redis-master" created
replicationcontroller "redis-master" created
service "redis-slave" created
replicationcontroller "redis-slave" created
service "frontend" created
replicationcontroller "frontend" created
```

You also can start the guestbook by running:

```console
$ kubectl create -f examples/guestbook/
```

Then, list all your services:

```console
$ kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR                            AGE
frontend          10.0.93.211      <none>            80/TCP        app=guestbook,tier=frontend         1h
redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master,tier=backend  1h
redis-slave       10.0.21.92       <none>            6379/TCP      app=redis,role=slave,tier=backend   1h
```

Now you can access the guestbook on each node with frontend service's `<ClusterIP>:Port`, e.g. `10.0.93.211:80` in this guide. `<ClusterIP>` is an a cluster-internal IP. If you want to access the guestbook from outside of the cluster, add `type: NodePort` to frontend service `spec` field. Then you can access the guestbook with `<NodeIP>:NodePort` from outside of the cluster. On cloud providers which support external load balancers, setting the type field to "LoadBalancer" will provision a load balancer for your Service. There are several ways for you to access the guestbook. You may learn from [Accessing services running on the cluster](../../docs/user-guide/accessing-the-cluster.md#accessing-services-running-on-the-cluster).

Clean up the guestbook:

```console
$ kubectl delete -f examples/guestbook/all-in-one/guestbook-all-in-one.yaml
```

or

```console
$ kubectl delete -f examples/guestbook/
```


### Step One: Start up the redis master

Before continuing to the gory details, we also recommend you to read [Quick walkthrough](../../docs/user-guide/#quick-walkthrough), [Thorough walkthough](../../docs/user-guide/#thorough-walkthrough) and [Concept guide](../../docs/user-guide/#concept-guide).
**Note**: The redis master in this example is *not* highly available.  Making it highly available would be an interesting, but intricate exercise— redis doesn't actually support multi-master deployments at this point in time, so high availability would be a somewhat tricky thing to implement, and might involve periodic serialization to disk, and so on.

#### Define a replication controller

To start the redis master, use the file `examples/guestbook/redis-master-controller.yaml`, which describes a single [pod](../../docs/user-guide/pods.md) running a redis key-value server in a container.

Although we have a single instance of our redis master, we are using a [replication controller](../../docs/user-guide/replication-controller.md) to enforce that exactly one pod keeps running. E.g., if the node were to go down, the replication controller will ensure that the redis master gets restarted on a healthy node. (In our simplified example, this could result in data loss.)

The file `examples/guestbook/redis-master-controller.yaml` defines the redis master replication controller:

<!-- BEGIN MUNGE: EXAMPLE redis-master-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: redis-master
  # these labels can be applied automatically 
  # from the labels in the pod template if not set
  labels:
    app: redis
    role: master
    tier: backend
spec:
  # this replicas value is default
  # modify it according to your case
  replicas: 1
  # selector can be applied automatically 
  # from the labels in the pod template if not set
  # selector:
  #   app: guestbook
  #   role: master
  #   tier: backend
  template:
    metadata:
      labels:
        app: redis
        role: master
        tier: backend
    spec:
      containers:
      - name: master
        image: redis
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
        ports:
        - containerPort: 6379
```

[Download example](redis-master-controller.yaml?raw=true)
<!-- END MUNGE: EXAMPLE redis-master-controller.yaml -->

#### Define a service

A Kubernetes [service](../../docs/user-guide/services.md) is a named load balancer that proxies traffic to one or more containers. This is done using the [labels](../../docs/user-guide/labels.md) metadata that we defined in the `redis-master` pod above.  As mentioned, we have only one redis master, but we nevertheless want to create a service for it. Why? Because it gives us a deterministic way to route to the single master using an elastic IP.

Services find the pods to load balance based on the pods' labels.
The selector field of the service description determines which pods will receive the traffic sent to the service, and the `port` and `targetPort` information defines what port the service proxy will run at.

The file `examples/guestbook/redis-master-service.yaml` defines the redis master service:

<!-- BEGIN MUNGE: EXAMPLE redis-master-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-master
  labels:
    app: redis
    role: master
    tier: backend
spec:
  ports:
    # the port that this service should serve on
  - port: 6379
    targetPort: 6379
  selector:
    app: redis
    role: master
    tier: backend
```

[Download example](redis-master-service.yaml?raw=true)
<!-- END MUNGE: EXAMPLE redis-master-service.yaml -->

#### Create a service

According to the [config best practices](../../docs/user-guide/config-best-practices.md), create a service before corresponding replication controllers so that the scheduler can spread the pods comprising the service. So we first create the service by running:

```console
$ kubectl create -f examples/guestbook/redis-master-service.yaml
service "redis-master" created
```

Then check the list of services, which should include the redis-master:

```console
$ kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR                            AGE
redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master,tier=backend  1h
```

This will cause all pods to see the redis master apparently running on <ip>:6379.  A service can map an incoming port to any `targetPort` in the backend pod.  Once created, the service proxy on each node is configured to set up a proxy on the specified port (in this case port 6379).

`targetPort` will default to `port` if it is omitted in the configuration. For simplicity's sake, we omit it in the following configurations.

The traffic flow from slaves to masters can be described in two steps, like so:

  - A *redis slave* will connect to "port" on the *redis master service*
  - Traffic will be forwarded from the service "port" (on the service node) to the  *targetPort* on the pod that the service listens to.

For more details, please see [Connecting applications](../../docs/user-guide/connecting-applications.md).

#### Finding a service

Kubernetes supports two primary modes of finding a service— environment variables and DNS.

The services in a Kubernetes cluster are discoverable inside other containers [via environment variables](../../docs/user-guide/services.md#environment-variables).

An alternative is to use the [cluster's DNS service](../../docs/user-guide/services.md#dns), if it has been enabled for the cluster.  This lets all pods do name resolution of services automatically, based on the service name.

This example has been configured to use the DNS service by default.

If your cluster does not have the DNS service enabled, then you can use environment variables by setting the
`GET_HOSTS_FROM` env value in both
`examples/guestbook/redis-slave-controller.yaml` and `examples/guestbook/frontend-controller.yaml`
from `dns` to `env` before you start up the app.
(However, this is unlikely to be necessary. You can check for the DNS service in the list of the clusters' services by
running `kubectl --namespace=kube-system get rc`, and looking for a controller prefixed `kube-dns`.)
Note that switching to env causes creation-order dependencies, since services need to be created before their clients that require env vars.

#### Create a replication controller

Second create the redis master pod in your Kubernetes cluster by running:

```console
$ kubectl create -f examples/guestbook/redis-master-controller.yaml
replicationcontroller "redis-master" created
```

You can see the replication controllers for your cluster by running:

```console
$ kubectl get rc
CONTROLLER                  CONTAINER(S)            IMAGE(S)                   SELECTOR                                      REPLICAS
redis-master                master                  redis                      app=redis,role=master,tier=backend            1
```

Then, you can list the pods in the cluster, to verify that the master is running:

```console
$ kubectl get pods
```

You'll see all pods in the cluster, including the redis master pod, and the status of each pod.
The name of the redis master will look similar to that in the following list:

```console
NAME                                READY     STATUS    RESTARTS   AGE
...
redis-master-dz33o                  1/1       Running   0          2h
```

(Note that an initial `docker pull` to grab a container image may take a few minutes, depending on network conditions. A pod will be reported as `Pending` while its image is being downloaded.)

`kubectl get pods` will show only the pods in the default [namespace](../../docs/user-guide/namespaces.md).  To see pods in all namespaces, run:

```
kubectl get pods --all-namespaces
```

For more details, please see [Configuring containers](../../docs/user-guide/configuring-containers.md) and [Deploying applications](../../docs/user-guide/deploying-applications.md).

#### Optional Interlude

You can get information about a pod, including the machine that it is running on, via `kubectl describe pods/<pod_name>`.  E.g., for the redis master, you should see something like the following (your pod name will be different):

```console
$ kubectl describe pods/redis-master-dz33o
...
Name:       redis-master-dz33o
Image(s):     redis
Node:       kubernetes-minion-krxw/10.240.67.201
Labels:       app=redis,role=master,tier=backend
Status:       Running
Replication Controllers:  redis-master (1/1 replicas created)
Containers:
  master:
    Image:    redis
    State:    Running
      Started:    Fri, 12 Jun 2015 12:53:46 -0700
    Ready:    True
    Restart Count:  0
Conditions:
  Type    Status
  Ready   True
No events.
```

The 'Node' is the name of the machine, e.g. `kubernetes-minion-krxw` in the example above.

If you want to view the container logs for a given pod, you can run:

```console
$ kubectl logs <pod_name>
```

These logs will usually give you enough information to troubleshoot.

However, if you should want to SSH to the listed host machine, you can inspect various logs there directly as well.  For example, with Google Compute Engine, using `gcloud`, you can SSH like this:

```console
me@workstation$ gcloud compute ssh kubernetes-minion-krxw
```

Then, you can look at the docker containers on the remote machine.  You should see something like this (the specifics of the IDs will be different):

```console
me@kubernetes-minion-krxw:~$ sudo docker ps
CONTAINER ID        IMAGE                                 COMMAND                 CREATED              STATUS              PORTS                   NAMES
...
0ffef9649265        redis:latest                          "/entrypoint.sh redi"   About a minute ago   Up About a minute                           k8s_master.869d22f3_redis-master-dz33o_default_1449a58a-5ead-11e5-a104-688f84ef8ef6_d74cb2b5
```

If you want to see the logs for a given container, you can run:

```console
$ docker logs <container_id>
```

### Step Two: Start up the redis slave

Now that the redis master is running, we can start up its 'read slaves'.

We'll define these as replicated pods as well, though this time— unlike for the redis master— we'll define the number of replicas to be 2.
In Kubernetes, a replication controller is responsible for managing multiple instances of a replicated pod. The replication controller will automatically launch new pods if the number of replicas falls below the specified number.
(This particular replicated pod is a great one to test this with -- you can try killing the docker processes for your pods directly, then watch them come back online on a new node shortly thereafter.)

Just like the master, we want to have a service to proxy connections to the redis slaves. In this case, in addition to discovery, the slave service will provide transparent load balancing to web app clients.

This time we put the service and RC into one [file](../../docs/user-guide/managing-deployments.md#organizing-resource-configurations). Group related objects together in a single file. This is often better than separate files.
The specification for the slaves is in `examples/guestbook/all-in-one/redis-slave.yaml`:

<!-- BEGIN MUNGE: EXAMPLE all-in-one/redis-slave.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-slave
  labels:
    app: redis
    role: slave
    tier: backend
spec:
  ports:
    # the port that this service should serve on
  - port: 6379
  selector:
    app: redis
    role: slave
    tier: backend
---
apiVersion: v1
kind: ReplicationController
metadata:
  name: redis-slave
  # these labels can be applied automatically
  # from the labels in the pod template if not set
  labels:
    app: redis
    role: slave
    tier: backend
spec:
  # this replicas value is default
  # modify it according to your case
  replicas: 2
  # selector can be applied automatically
  # from the labels in the pod template if not set
  # selector:
  #   app: guestbook
  #   role: slave
  #   tier: backend
  template:
    metadata:
      labels:
        app: redis
        role: slave
        tier: backend
    spec:
      containers:
      - name: slave
        image: gcr.io/google_samples/gb-redisslave:v1
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
        env:
        - name: GET_HOSTS_FROM
          value: dns
          # If your cluster config does not include a dns service, then to
          # instead access an environment variable to find the master
          # service's host, comment out the 'value: dns' line above, and
          # uncomment the line below.
          # value: env
        ports:
        - containerPort: 6379
```

[Download example](all-in-one/redis-slave.yaml?raw=true)
<!-- END MUNGE: EXAMPLE all-in-one/redis-slave.yaml -->

This time the selector for the service is `app=redis,role=slave,tier=backend`, because that identifies the pods running redis slaves. It is generally helpful to set labels on your service itself as we've done here to make it easy to locate them with the `kubectl get services -l "app=redis,role=slave,tier=backend"` command. More lables usage, see [using-labels-effectively](../../docs/user-guide/managing-deployments.md#using-labels-effectively).

Now that you have created the specification, create it in your cluster by running:

```console
$ kubectl create -f examples/guestbook/all-in-one/redis-slave.yaml
service "redis-slave" created
replicationcontroller "redis-slave" created

$ kubectl get services
NAME             CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR                            AGE
redis-master     10.0.136.3       <none>            6379/TCP      app=redis,role=master,tier=backend  1h
redis-slave      10.0.21.92       <none>            6379/TCP      app=redis,role=slave,tier=backend   1h

$ kubectl get rc
CONTROLLER                  CONTAINER(S)            IMAGE(S)                                 SELECTOR                                      REPLICAS
redis-master                master                  redis                                    app=redis,role=master,tier=backend            1
redis-slave                 slave                   gcr.io/google_samples/gb-redisslave:v1   app=redis,role=slave,tier=backend             2
```

Once the replication controller is up, you can list the pods in the cluster, to verify that the master and slaves are running.  You should see a list that includes something like the following:

```console
$ kubectl get pods
NAME                                READY     STATUS    RESTARTS   AGE
...
redis-master-dz33o                  1/1       Running   0          2h
redis-slave-35mer                   1/1       Running   0          2h
redis-slave-iqkhy                   1/1       Running   0          2h
```

You should see a single redis master pod and two redis slave pods.  As mentioned above, you can get more information about any pod with: `kubectl describe pods/<pod_name>`. And also can view the resources on [kube-ui](../../docs/user-guide/ui.md).

### Step Three: Start up the guestbook frontend

A frontend pod is a simple PHP server that is configured to talk to either the slave or master services, depending on whether the client request is a read or a write. It exposes a simple AJAX interface, and serves an Angular-based UX.
Again we'll create a set of replicated frontend pods instantiated by a replication controller— this time, with three replicas.

As with the other pods, we now want to create a service to group the frontend pods.
The RC and service are described in the file `frontend.yaml`:

<!-- BEGIN MUNGE: EXAMPLE all-in-one/frontend.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  name: frontend
  labels:
    app: guestbook
    tier: frontend
spec:
  # if your cluster supports it, uncomment the following to automatically create
  # an external load-balanced IP for the frontend service.
  # type: LoadBalancer
  ports:
    # the port that this service should serve on
  - port: 80
  selector:
    app: guestbook
    tier: frontend
---
apiVersion: v1
kind: ReplicationController
metadata:
  name: frontend
  # these labels can be applied automatically
  # from the labels in the pod template if not set
  labels:
    app: guestbook
    tier: frontend
spec:
  # this replicas value is default
  # modify it according to your case
  replicas: 3
  # selector can be applied automatically
  # from the labels in the pod template if not set
  # selector:
  #   app: guestbook
  #   tier: frontend
  template:
    metadata:
      labels:
        app: guestbook
        tier: frontend
    spec:
      containers:
      - name: php-redis
        image: gcr.io/google_samples/gb-frontend:v3
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
        env:
        - name: GET_HOSTS_FROM
          value: dns
          # If your cluster config does not include a dns service, then to
          # instead access environment variables to find service host
          # info, comment out the 'value: dns' line above, and uncomment the
          # line below.
          # value: env
        ports:
        - containerPort: 80
```

[Download example](all-in-one/frontend.yaml?raw=true)
<!-- END MUNGE: EXAMPLE all-in-one/frontend.yaml -->

#### Using 'type: LoadBalancer' for the frontend service (cloud-provider-specific)

For supported cloud providers, such as Google Compute Engine or Google Container Engine, you can specify to use an external load balancer
in the service `spec`, to expose the service onto an external load balancer IP.
To do this, uncomment the `type: LoadBalancer` line in the `frontend.yaml` file before you start the service.

[See the appendix below](#appendix-accessing-the-guestbook-site-externally) on accessing the guestbook site externally for more details.

Create the service and replication controller like this:

```console
$ kubectl create -f examples/guestbook/all-in-one/frontend.yaml
service "frontend" created
replicationcontroller "frontend" created
```

Then, list all your services again:

```console
$ kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR                            AGE
frontend          10.0.93.211      <none>            80/TCP        app=guestbook,tier=frontend         1h
redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master,tier=backend  1h
redis-slave       10.0.21.92       <none>            6379/TCP      app=redis,role=slave,tier=backend   1h
```

Also list all your replication controllers:

```console
$ kubectl get rc
CONTROLLER                  CONTAINER(S)            IMAGE(S)                                   SELECTOR                               REPLICAS
frontend                    php-redis               kubernetes/example-guestbook-php-redis:v3  app=guestbook,tier=frontend            3
redis-master                master                  redis                                      app=redis,role=master,tier=backend     1
redis-slave                 slave                   gcr.io/google_samples/gb-redisslave:v1     app=redis,role=slave,tier=backend      2
```

Once it's up (again, it may take up to thirty seconds to create the pods) you can list the pods with specified labels the cluster, to verify that the master, slaves and frontends are all running. You should see a list contains pods with label tier like the following:

```console
$ kubectl get pods -L tier
NAME                                READY     STATUS    RESTARTS   AGE     TIER
frontend-4o11g                      1/1       Running   0          2h      frontend
frontend-u9aq6                      1/1       Running   0          2h      frontend
frontend-yga1l                      1/1       Running   0          2h      frontend
redis-master-dz33o                  1/1       Running   0          2h      backend
redis-slave-35mer                   1/1       Running   0          2h      backend
redis-slave-iqkhy                   1/1       Running   0          2h      backend
```

You should see a single redis master pod, two redis slaves, and three frontend pods.

The code for the PHP server that the frontends are running is in `examples/guestbook/php-redis/guestbook.php`.  It looks like this:

```php
<?

set_include_path('.:/usr/local/lib/php');

error_reporting(E_ALL);
ini_set('display_errors', 1);

require 'Predis/Autoloader.php';

Predis\Autoloader::register();

if (isset($_GET['cmd']) === true) {
  $host = 'redis-master';
  if (getenv('GET_HOSTS_FROM') == 'env') {
    $host = getenv('REDIS_MASTER_SERVICE_HOST');
  }
  header('Content-Type: application/json');
  if ($_GET['cmd'] == 'set') {
    $client = new Predis\Client([
      'scheme' => 'tcp',
      'host'   => $host,
      'port'   => 6379,
    ]);

    $client->set($_GET['key'], $_GET['value']);
    print('{"message": "Updated"}');
  } else {
    $host = 'redis-slave';
    if (getenv('GET_HOSTS_FROM') == 'env') {
      $host = getenv('REDIS_SLAVE_SERVICE_HOST');
    }
    $client = new Predis\Client([
      'scheme' => 'tcp',
      'host'   => $host,
      'port'   => 6379,
    ]);

    $value = $client->get($_GET['key']);
    print('{"data": "' . $value . '"}');
  }
} else {
  phpinfo();
} ?>
```

Note the use of the `redis-master` and `redis-slave` host names-- we're finding those services via the Kubernetes cluster's DNS service, as discussed above.  All the frontend replicas will write to the load-balancing redis-slaves service, which can be highly replicated as well.

### Step Four: Cleanup

If you are in a live kubernetes cluster, you can just kill the pods by deleteing the replication controllers and services. Using labels to select the resources to delete is an easy way to do this in one command.

```console
$ kubectl delete rc -l "app in (redis, guestbook)"
$ kubectl delete service -l "app in (redis, guestbook)"
```

To completely tear down a Kubernetes cluster, if you ran this from source, you can use:

```console
$ <kubernetes>/cluster/kube-down.sh
```

### Troubleshooting

If you are having trouble bringing up your guestbook app, double check that your external IP is properly defined for your frontend service, and that the firewall for your cluster nodes is open to port 80.

Then, see the [troubleshooting documentation](../../docs/troubleshooting.md) for a further list of common issues and how you can diagnose them.



### Appendix: Accessing the guestbook site externally

You'll want to set up your guestbook service so that it can be accessed from outside of the internal Kubernetes network. Above, we introduced one way to do that, using the `type: LoadBalancer` spec.

More generally, Kubernetes supports two ways of exposing a service onto an external IP address: `NodePort`s and `LoadBalancer`s , as described [here](../../docs/user-guide/services.md#publishing-services---service-types).

If the `LoadBalancer` specification is used, it can take a short period for an external IP to show up in `kubectl get services` output, but you should shortly see it listed as well, e.g. like this:

```console
$ kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR                             AGE
frontend          10.0.93.211      130.211.188.51    80/TCP        app=guestbook,tier=frontend          1h
redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master,tier=backend   1h
redis-slave       10.0.21.92       <none>            6379/TCP      app=redis,role=master,tier=backend   1h
```

Once you've exposed the service to an external IP, visit the IP to see your guestbook in action. E.g., `http://130.211.188.51:80` in the example above.

You should see a web page that looks something like this (without the messages).  Try adding some entries to it!

<img width="50%" src="http://amy-jo.storage.googleapis.com/images/gb_k8s_ex1.png">

If you are more advanced in the ops arena, you can also manually get the service IP from looking at the output of `kubectl get pods,services`, and modify your firewall using standard tools and services (firewalld, iptables, selinux) which you are already familiar with.

#### Google Compute Engine External Load Balancer Specifics

In Google Compute Engine, Kubernetes automatically creates forwarding rule for services with `LoadBalancer`.

You can list the forwarding rules like this.  The forwarding rule also indicates the external IP.

```console
$ gcloud compute forwarding-rules list
NAME                  REGION      IP_ADDRESS     IP_PROTOCOL TARGET
frontend              us-central1 130.211.188.51 TCP         us-central1/targetPools/frontend
```

In Google Compute Engine, you also may need to open the firewall for port 80 using the [console][cloud-console] or the `gcloud` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion` (replace with your tags as appropriate):

```console
$ gcloud compute firewall-rules create --allow=tcp:80 --target-tags=kubernetes-minion kubernetes-minion-80
```

For GCE kubernetes startup details, see the [Getting started on Google Compute Engine](../../docs/getting-started-guides/gce.md)

For Google Compute Engine details about limiting traffic to specific sources, see the [Google Compute Engine firewall documentation][gce-firewall-docs].

[cloud-console]: https://console.developer.google.com
[gce-firewall-docs]: https://cloud.google.com/compute/docs/networking#firewalls

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/guestbook/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
