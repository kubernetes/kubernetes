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
[here](http://releases.k8s.io/release-1.0/examples/guestbook/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->


## Guestbook Example

This example shows how to build a simple, multi-tier web application using Kubernetes and [Docker](https://www.docker.com/).

**Table of Contents**

  - [Step Zero: Prerequisites](#step-zero-prerequisites)
  - [Step One: Start up the redis master](#step-one-start-up-the-redis-master)
    - [Optional Interlude](#optional-interlude)
  - [Step Two: Fire up the redis master service](#step-two-fire-up-the-redis-master-service)
    - [Finding a service](#finding-a-service)
  - [Step Three: Fire up the replicated slave pods](#step-three-fire-up-the-replicated-slave-pods)
  - [Step Four: Create the redis slave service](#step-four-create-the-redis-slave-service)
  - [Step Five: Create the frontend replicated pods](#step-five-create-the-frontend-replicated-pods)
  - [Step Six: Set up the guestbook frontend service.](#step-six-set-up-the-guestbook-frontend-service)
    - [Using 'type: LoadBalancer' for the frontend service (cloud-provider-specific)](#using-type-loadbalancer-for-the-frontend-service-cloud-provider-specific)
    - [Create the Frontend Service](#create-the-frontend-service)
    - [Accessing the guestbook site externally](#accessing-the-guestbook-site-externally)
      - [Google Compute Engine External Load Balancer Specifics](#gce-external-load-balancer-specifics)
  - [Step Seven: Cleanup](#step-seven-cleanup)
  - [Troubleshooting](#troubleshooting)

The example consists of:

- A web frontend
- A [redis](http://redis.io/) master (for storage), and a replicated set of redis 'slaves'.

The web front end interacts with the redis master via javascript redis API calls.

**Note**:  If you are running this example on a [Google Container Engine](https://cloud.google.com/container-engine/) installation, see [this Container Engine guestbook walkthrough](https://cloud.google.com/container-engine/docs/tutorials/guestbook) instead. The basic concepts are the same, but the walkthrough is tailored to a Container Engine setup.

### Step Zero: Prerequisites

This example requires a running Kubernetes cluster.  See the [Getting Started guides](../../docs/getting-started-guides/) for how to get started. As noted above, if you have a Google Container Engine cluster set up, go [here](https://cloud.google.com/container-engine/docs/tutorials/guestbook) instead.

### Step One: Start up the redis master

**Note**: The redis master in this example is *not* highly available.  Making it highly available would be an interesting, but intricate exercise— redis doesn't actually support multi-master deployments at this point in time, so high availability would be a somewhat tricky thing to implement, and might involve periodic serialization to disk, and so on.

To start the redis master, use the file `examples/guestbook/redis-master-controller.yaml`, which describes a single [pod](../../docs/user-guide/pods.md) running a redis key-value server in a container.

Although we have a single instance of our redis master, we are using a [replication controller](../../docs/user-guide/replication-controller.md) to enforce that exactly one pod keeps running. E.g., if the node were to go down, the replication controller will ensure that the redis master gets restarted on a healthy node. (In our simplified example, this could result in data loss.)

<!-- BEGIN MUNGE: EXAMPLE redis-master-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: redis-master
  labels:
    name: redis-master
spec:
  replicas: 1
  selector:
    name: redis-master
  template:
    metadata:
      labels:
        name: redis-master
    spec:
      containers:
      - name: master
        image: redis
        ports:
        - containerPort: 6379
```

[Download example](redis-master-controller.yaml)
<!-- END MUNGE: EXAMPLE redis-master-controller.yaml -->

Change to the `<kubernetes>/examples/guestbook` directory if you're not already there. Create the redis master pod in your Kubernetes cluster by running:

```console
$ kubectl create -f examples/guestbook/redis-master-controller.yaml
replicationcontrollers/redis-master
```

The `replicationcontrollers/redis-master` line is the expected response to this operation.
You can see the replication controllers for your cluster by running:

```console
$ kubectl get rc
CONTROLLER                             CONTAINER(S)            IMAGE(S)                                 SELECTOR                     REPLICAS
redis-master                           master                  redis                                    name=redis-master            1
```

Then, you can list the pods in the cluster, to verify that the master is running:

```console
$ kubectl get pods
```

You'll see all pods in the cluster, including the redis master pod, and the status of each pod.
The name of the redis master will look similar to that in the following list:

```console
NAME                                           READY     STATUS    RESTARTS   AGE
...
redis-master-dz33o                             1/1       Running   0          2h
```

(Note that an initial `docker pull` to grab a container image may take a few minutes, depending on network conditions. A pod will be reported as `Pending` while its image is being downloaded.)

#### Optional Interlude

You can get information about a pod, including the machine that it is running on, via `kubectl describe pods/<pod_name>`.  E.g., for the redis master, you should see something like the following (your pod name will be different):

```console
$ kubectl describe pods/redis-master-dz33o
...
Name:       redis-master-dz33o
Image(s):     redis
Node:       kubernetes-minion-krxw/10.240.67.201
Labels:       name=redis-master
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
CONTAINER ID        IMAGE                                  COMMAND                CREATED              STATUS              PORTS                    NAMES
...
0ffef9649265        redis:latest                           "redis-server /etc/r"   About a minute ago   Up About a minute                            k8s_redis-master.767aef46_redis-master-controller-gb50a.default.api_4530d7b3-ae5d-11e4-bf77-42010af0d719_579ee964
```

If you want to see the logs for a given container, you can run:

```console
$ docker logs <container_id>
```

### Step Two: Fire up the redis master service

A Kubernetes [service](../../docs/user-guide/services.md) is a named load balancer that proxies traffic to one or more containers. This is done using the [labels](../../docs/user-guide/labels.md) metadata that we defined in the `redis-master` pod above.  As mentioned, we have only one redis master, but we nevertheless want to create a service for it.  Why?  Because it gives us a deterministic way to route to the single master using an elastic IP.

Services find the pods to load balance based on the pods' labels.
The pod that you created in [Step One](#step-one-start-up-the-redis-master) has the label `name=redis-master`.
The selector field of the service description determines which pods will receive the traffic sent to the service, and the `port` and `targetPort` information defines what port the service proxy will run at.

The file `examples/guestbook/redis-master-service.yaml` defines the redis master service:

<!-- BEGIN MUNGE: EXAMPLE redis-master-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-master
  labels:
    name: redis-master
spec:
  ports:
    # the port that this service should serve on
  - port: 6379
    targetPort: 6379
  selector:
    name: redis-master
```

[Download example](redis-master-service.yaml)
<!-- END MUNGE: EXAMPLE redis-master-service.yaml -->

Create the service by running:

```console
$ kubectl create -f examples/guestbook/redis-master-service.yaml
services/redis-master
```

Then check the list of services, which should include the redis-master:

```console
$ kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master  1h
...
```

This will cause all pods to see the redis master apparently running on <ip>:6379.  A service can map an incoming port to any `targetPort` in the backend pod.  Once created, the service proxy on each node is configured to set up a proxy on the specified port (in this case port 6379).

`targetPort` will default to `port` if it is omitted in the configuration. For simplicity's sake, we omit it in the following configurations.

The traffic flow from slaves to masters can be described in two steps, like so:

  - A *redis slave* will connect to "port" on the *redis master service*
  - Traffic will be forwarded from the service "port" (on the service node) to the  *targetPort* on the pod that the service listens to.

#### Finding a service

Kubernetes supports two primary modes of finding a service— environment variables and DNS.

The services in a Kubernetes cluster are discoverable inside other containers [via environment variables](../../docs/user-guide/services.md#environment-variables).

An alternative is to use the [cluster's DNS service](../../docs/user-guide/services.md#dns), if it has been enabled for the cluster.  This lets all pods do name resolution of services automatically, based on the service name.
We'll use the DNS service for this example.  E.g., you can see the service name, `redis-master`, accessed as a `host` value in the PHP script in [Step 5](#step-five-create-the-frontend-replicated-pods).

**Note**: **If your cluster does not have the DNS service enabled, then this example will not work out of the box.** You will need to edit `examples/guestbook/php-redis/index.php` to use environment variables for service discovery instead, then rebuild the container image from the `Dockerfile` in that directory.  (However, this is unlikely to be necessary. You can check for the DNS service in the list of the clusters' services.)


### Step Three: Fire up the replicated slave pods

Now that the redis master is running, we can start up its 'read slaves'.

We'll define these as replicated pods as well, though this time— unlike for the redis master— we'll define the number of replicas to be 2.
In Kubernetes, a replication controller is responsible for managing multiple instances of a replicated pod.  The replication controller will automatically launch new pods if the number of replicas falls below the specified number.
(This particular replicated pod is a great one to test this with -- you can try killing the docker processes for your pods directly, then watch them come back online on a new node shortly thereafter.)

To create the replicated pod, use the file `examples/guestbook/redis-slave-controller.yaml`, which looks like this:

<!-- BEGIN MUNGE: EXAMPLE redis-slave-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: redis-slave
  labels:
    name: redis-slave
spec:
  replicas: 2
  selector:
    name: redis-slave
  template:
    metadata:
      labels:
        name: redis-slave
    spec:
      containers:
      - name: worker
        image: kubernetes/redis-slave:v2
        ports:
        - containerPort: 6379
```

[Download example](redis-slave-controller.yaml)
<!-- END MUNGE: EXAMPLE redis-slave-controller.yaml -->

and create the replication controller by running:

```console
$ kubectl create -f examples/guestbook/redis-slave-controller.yaml
replicationcontrollers/redis-slave

$ kubectl get rc
CONTROLLER                             CONTAINER(S)            IMAGE(S)                                 SELECTOR                     REPLICAS
redis-master                           master                  redis                                    name=redis-master            1
redis-slave                            slave                   kubernetes/redis-slave:v2                name=redis-slave             2
```

Once the replication controller is up, you can list the pods in the cluster, to verify that the master and slaves are running.  You should see a list that includes something like the following:

```console
$ kubectl get pods
NAME                                           READY     STATUS    RESTARTS   AGE
...
redis-master-dz33o                             1/1       Running   0          2h
redis-slave-35mer                              1/1       Running   0          2h
redis-slave-iqkhy                              1/1       Running   0          2h
```

You should see a single redis master pod and two redis slave pods.  As mentioned above, you can get more information about any pod with: `kubectl describe pods/<pod_name>`.

### Step Four: Create the redis slave service

Just like the master, we want to have a service to proxy connections to the redis slaves. In this case, in addition to discovery, the slave service will provide transparent load balancing to web app clients.

The service specification for the slaves is in `examples/guestbook/redis-slave-service.yaml`:

<!-- BEGIN MUNGE: EXAMPLE redis-slave-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-slave
  labels:
    name: redis-slave
spec:
  ports:
    # the port that this service should serve on
  - port: 6379
  selector:
    name: redis-slave
```

[Download example](redis-slave-service.yaml)
<!-- END MUNGE: EXAMPLE redis-slave-service.yaml -->

This time the selector for the service is `name=redis-slave`, because that identifies the pods running redis slaves. It may also be helpful to set labels on your service itself as we've done here to make it easy to locate them with the `kubectl get services -l "label=value"` command.

Now that you have created the service specification, create it in your cluster by running:

```console
$ kubectl create -f examples/guestbook/redis-slave-service.yaml
services/redis-slave

$ kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master  1h
redis-slave       10.0.21.92       <none>            6379/TCP      app-redis,role=slave   1h
```

### Step Five: Create the frontend replicated pods

<a href="#step-five-create-the-frontend-replicated-pods"></a>

A frontend pod is a simple PHP server that is configured to talk to either the slave or master services, depending on whether the client request is a read or a write. It exposes a simple AJAX interface, and serves an Angular-based UX.
Again we'll create a set of replicated frontend pods instantiated by a replication controller— this time, with three replicas.

The pod is described in the file `examples/guestbook/frontend-controller.yaml`:

<!-- BEGIN MUNGE: EXAMPLE frontend-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: frontend
  labels:
    name: frontend
spec:
  replicas: 3
  selector:
    name: frontend
  template:
    metadata:
      labels:
        name: frontend
    spec:
      containers:
      - name: php-redis
        image: kubernetes/example-guestbook-php-redis:v2
        ports:
        - containerPort: 80
```

[Download example](frontend-controller.yaml)
<!-- END MUNGE: EXAMPLE frontend-controller.yaml -->

Using this file, you can turn up your frontend with:

```console
$ kubectl create -f examples/guestbook/frontend-controller.yaml
replicationcontrollers/frontend
```

Then, list all your replication controllers:

```console
$ kubectl get rc
CONTROLLER                             CONTAINER(S)            IMAGE(S)                                   SELECTOR                     REPLICAS
frontend                               php-redis               kubernetes/example-guestbook-php-redis:v2  name=frontend                3
redis-master                           master                  redis                                      name=redis-master            1
redis-slave                            slave                   kubernetes/redis-slave:v2                  name=redis-slave             2
```

Once it's up (again, it may take up to thirty seconds to create the pods) you can list the pods in the cluster, to verify that the master, slaves and frontends are all running.  You should see a list that includes something like the following:

```console
$ kubectl get pods
NAME                                           READY     STATUS    RESTARTS   AGE
...
frontend-4o11g                                 1/1       Running   0          2h
frontend-u9aq6                                 1/1       Running   0          2h
frontend-yga1l                                 1/1       Running   0          2h
...
redis-master-dz33o                             1/1       Running   0          2h
redis-slave-35mer                              1/1       Running   0          2h
redis-slave-iqkhy                              1/1       Running   0          2h
```

You should see a single redis master pod, two redis slaves, and three frontend pods.

The code for the PHP server that the frontends are running looks like this:

```php
<?

set_include_path('.:/usr/share/php:/usr/share/pear:/vendor/predis');

error_reporting(E_ALL);
ini_set('display_errors', 1);

require 'predis/autoload.php';

if (isset($_GET['cmd']) === true) {
  header('Content-Type: application/json');
  if ($_GET['cmd'] == 'set') {
    $client = new Predis\Client([
      'scheme' => 'tcp',
      'host'   => 'redis-master',
      'port'   => 6379,
    ]);

    $client->set($_GET['key'], $_GET['value']);
    print('{"message": "Updated"}');
  } else {
    $client = new Predis\Client([
      'scheme' => 'tcp',
      'host'   => 'redis-slave',
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


### Step Six: Set up the guestbook frontend service.

As with the other pods, we now want to create a service to group your frontend pods.
The service is described in the file `frontend-service.yaml`:

<!-- BEGIN MUNGE: EXAMPLE frontend-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  name: frontend
  labels:
    name: frontend
spec:
  # if your cluster supports it, uncomment the following to automatically create
  # an external load-balanced IP for the frontend service.
  # type: LoadBalancer
  ports:
    # the port that this service should serve on
    - port: 80
  selector:
    name: frontend
```

[Download example](frontend-service.yaml)
<!-- END MUNGE: EXAMPLE frontend-service.yaml -->

#### Using 'type: LoadBalancer' for the frontend service (cloud-provider-specific)

For supported cloud providers, such as Google Compute Engine or Google Container Engine, you can specify to use an external load balancer
in the service `spec`, to expose the service onto an external load balancer IP.
To do this, uncomment the `type: LoadBalancer` line in the `frontend-service.yaml` file before you start the service.

[See the section below](#accessing-the-guestbook-site-externally) on accessing the guestbook site externally for more details.


#### Create the Frontend Service ####

Create the service like this:

```console
$ kubectl create -f examples/guestbook/frontend-service.yaml
services/frontend
```

Then, list all your services again:

```console
$ kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
frontend          10.0.93.211      <none>            80/TCP        name=frontend          1h
redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master  1h
redis-slave       10.0.21.92       <none>            6379/TCP      app-redis,role=slave   1h
```


#### Accessing the guestbook site externally

<a href="#accessing-the-guestbook-site-externally"></a>

You'll want to set up your guestbook service so that it can be accessed from outside of the internal Kubernetes network. Above, we introduced one way to do that, using the `type: LoadBalancer` spec.

More generally, Kubernetes supports two ways of exposing a service onto an external IP address: `NodePort`s and `LoadBalancer`s , as described [here](../../docs/user-guide/services.md#external-services).

If the `LoadBalancer` specification is used, it can take a short period for an external IP to show up in `kubectl get services` output, but you should shortly see it listed as well, e.g. like this:

```console
$ kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
frontend          10.0.93.211      130.211.188.51    80/TCP        name=frontend          1h
redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master  1h
redis-slave       10.0.21.92       <none>            6379/TCP      app-redis,role=slave   1h
```

Once you've exposed the service to an external IP, visit the IP to see your guestbook in action. E.g., `http://130.211.188.51:80` in the example above.

You should see a web page that looks something like this (without the messages).  Try adding some entries to it!

<img width="50%" src="http://amy-jo.storage.googleapis.com/images/gb_k8s_ex1.png">

If you are more advanced in the ops arena, you can also manually get the service IP from looking at the output of `kubectl get pods,services`, and modify your firewall using standard tools and services (firewalld, iptables, selinux) which you are already familiar with.

##### Google Compute Engine External Load Balancer Specifics

In Google Compute Engine, `kubectl` automatically creates forwarding rule for services with `LoadBalancer`.

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

For Google Compute Engine details about limiting traffic to specific sources, see the [Google Compute Engine firewall documentation][gce-firewall-docs].

[cloud-console]: https://console.developer.google.com
[gce-firewall-docs]: https://cloud.google.com/compute/docs/networking#firewalls

### Step Seven: Cleanup

If you are in a live kubernetes cluster, you can just kill the pods by stopping the replication controllers and deleting the services.  Using labels to select the resources to stop or delete is an easy way to do this in one command.

```console
kubectl stop rc -l "name in (redis-master, redis-slave, frontend)"
kubectl delete service -l "name in (redis-master, redis-slave, frontend)"
```

To completely tear down a Kubernetes cluster, if you ran this from source, you can use:

```console
$ <kubernetes>/cluster/kube-down.sh
```

### Troubleshooting

If you are having trouble bringing up your guestbook app, double check that your external IP is properly defined for your frontend service, and that the firewall for your cluster nodes is open to port 80.

Then, see the [troubleshooting documentation](../../docs/troubleshooting.md) for a further list of common issues and how you can diagnose them.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/guestbook/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
