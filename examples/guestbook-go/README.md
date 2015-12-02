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
[here](http://releases.k8s.io/release-1.1/examples/guestbook-go/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Guestbook Example

This example shows how to build a simple multi-tier web application using Kubernetes and Docker. The application consists of a web front-end, Redis master for storage, and replicated set of Redis slaves, all for which we will create Kubernetes replication controllers, pods, and services.

If you are running a cluster in Google Container Engine (GKE), instead see the [Guestbook Example for Google Container Engine](https://cloud.google.com/container-engine/docs/tutorials/guestbook).

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

  - [Guestbook Example](#guestbook-example)
    - [Prerequisites](#prerequisites)
    - [Quick Start](#quick-start)
    - [Step one: Create the Redis master](#step-one-create-the-redis-master)
      - [Create a service](#create-a-service)
      - [Create a replication controller](#create-a-replication-controller)
    - [Step Two: Create the Redis slave](#step-two-create-the-redis-slave)
    - [Step Three: Create the guestbook](#step-three-create-the-guestbook)
    - [Step Four: View the guestbook](#step-four-view-the-guestbook)
    - [Step Five: Cleanup](#step-five-cleanup)
    - [Troubleshooting](#troubleshooting)

<!-- END MUNGE: GENERATED_TOC -->

### Prerequisites

This example requires a running Kubernetes cluster. See the [Getting Started guides](../../docs/getting-started-guides/) for how to get started. And follow the [Prerequisites](../../docs/user-guide/prereqs.md) to make sure your `kubectl` is ok.

**Tip:** View all the `kubectl` commands, including their options and descriptions in the [kubectl CLI reference](../../docs/user-guide/kubectl/kubectl.md).

### Quick Start

This section shows a simplest way to get the example work. If you want to know the details, you should skip this and read [the rest of the example](#step-one-create-the-redis-master).

Start the guestbook with one command:

```console
$ kubectl create -f examples/guestbook-go/all-in-one/guestbook-go-all-in-one.yaml
service "redis-master" created
replicationcontroller "redis-master" created
service "redis-slave" created
replicationcontroller "redis-slave" created
service "guestbook" created
replicationcontroller "guestbook" created
```

You also can start the guestbook by running:

```console
$ kubectl create -f examples/guestbook-go/
```

Then, list all your services:

```console
$ kubectl get services
    NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
    guestbook         10.0.217.218     146.148.81.8      3000/TCP      app=guestbook          1h
    redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master  1h
    redis-slave       10.0.21.92       <none>            6379/TCP      app-redis,role=slave   1h
```

Now you can access the guestbook on each node with frontend service's `<ClusterIP>:Port`, e.g. `10.0.217.218:3000` in this guide. `<ClusterIP>` is a cluster-internal IP. If you want to access the guestbook from outside of the cluster, add `type: NodePort` to frontend service `spec` field. Then you can access the guestbook with `<NodeIP>:NodePort` from outside of the cluster. On cloud providers which support external load balancers, setting the type field to "LoadBalancer" will provision a load balancer for your Service, e.g. `146.148.81.8:3000` in this guide. There are several ways for you to access the guestbook. You may learn from [Accessing services running on the cluster](../../docs/user-guide/accessing-the-cluster.md#accessing-services-running-on-the-cluster).

Clean up the guestbook:

```console
$ kubectl delete -f examples/guestbook-go/all-in-one/guestbook-go-all-in-one.yaml
```

or

```console
$ kubectl delete -f examples/guestbook-go/
```

### Step one: Create the Redis master

A Kubernetes '[service](../../docs/user-guide/services.md)' is a named load balancer that proxies traffic to one or more pods. The services in a Kubernetes cluster are discoverable inside other pods via environment variables or DNS.

Services find the pods to load balance based on pod labels. The pod that you created in Step One has the label `app=redis` and `role=master`. The selector field of the service determines which pods will receive the traffic sent to the service. According to the [config best practices](../../docs/user-guide/config-best-practices.md), create a service before corresponding replication controllers so that the scheduler can spread the pods comprising the service. So we create the service first.

#### Create a service

1. Use the [redis-master-service.yaml](redis-master-service.yaml) file to create the service in your Kubernetes cluster by running the `kubectl create -f` *`filename`* command:

    ```console
    $ kubectl create -f examples/guestbook-go/redis-master-service.yaml
    service "redis-master" created
    ```

2. To verify that the redis-master service is up, list the services you created in the cluster with the `kubectl get services` command:

    ```console
    $ kubectl get services
    NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
    redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master  1h
    ...
    ```

    Result: All new pods will see the `redis-master` service running on the host (`$REDIS_MASTER_SERVICE_HOST` environment variable) at port 6379, or running on `redis-master:6379`. After the service is created, the service proxy on each node is configured to set up a proxy on the specified port (in our example, that's port 6379).

For more details, please see [Connecting applications](../../docs/user-guide/connecting-applications.md).

#### Create a replication controller

Use the `examples/guestbook-go/redis-master-controller.yaml` file to create a [replication controller](../../docs/user-guide/replication-controller.md) and Redis master [pod](../../docs/user-guide/pods.md). The pod runs a Redis key-value server in a container. Using a replication controller is the preferred way to launch long-running pods, even for 1 replica, so that the pod benefits from the self-healing mechanism in Kubernetes (keeps the pods alive).

1. Use the [redis-master-controller.yaml](redis-master-controller.yaml) file to create the Redis master replication controller in your Kubernetes cluster by running the `kubectl create -f` *`filename`* command:

    ```console
    $ kubectl create -f examples/guestbook-go/redis-master-controller.yaml
    replicationcontroller "redis-master" create
    ```

2. To verify that the redis-master controller is up, list the replication controllers you created in the cluster with the `kubectl get rc` command(if you don't specify a `--namespace`, the `default` namespace will be used. The same below):

    ```console
    $ kubectl get rc
    CONTROLLER             CONTAINER(S)            IMAGE(S)                    SELECTOR                         REPLICAS
    redis-master           redis-master            gurpartap/redis             app=redis,role=master            1
    ...
    ```

    Result: The replication controller then creates the single Redis master pod.

3. To verify that the redis-master pod is running, list the pods you created in cluster with the `kubectl get pods` command:

    ```console
    $ kubectl get pods
    NAME                        READY     STATUS    RESTARTS   AGE
    redis-master-xx4uv          1/1       Running   0          1m
    ...
    ```

    Result: You'll see a single Redis master pod and the machine where the pod is running after the pod gets placed (may take up to thirty seconds).

4. To verify what containers are running in the redis-master pod, you can SSH to that machine with `gcloud compute ssh --zone` *`zone_name`* *`host_name`* and then run `docker ps`:

    ```console
    me@workstation$ gcloud compute ssh --zone us-central1-b kubernetes-minion-bz1p
    
    me@kubernetes-minion-3:~$ sudo docker ps
    CONTAINER ID        IMAGE                      COMMAND                CREATED             STATUS
    d5c458dabe50        gurpartap/redis:latest     "/usr/local/bin/redi   5 minutes ago       Up 5 minutes
    ```

    Note: The initial `docker pull` can take a few minutes, depending on network conditions.

For more details, please see [Configuring containers](../../docs/user-guide/configuring-containers.md) and [Deploying applications](../../docs/user-guide/deploying-applications.md).

### Step Two: Create the Redis slave

Just like the master, we want to have a service to proxy connections to the read slaves. In this case, in addition to discovery, the Redis slave service provides transparent load balancing to clients.

1. Use the [redis-slave-service.yaml](redis-slave-service.yaml) file to create the Redis slave service by running the `kubectl create -f` *`filename`* command:

    ```console
    $ kubectl create -f examples/guestbook-go/redis-slave-service.yaml
    service "redis-slave" created
    ```

2. To verify that the redis-slave service is up, list the services you created in the cluster with the `kubectl get services` command:

    ```console
    $ kubectl get services
    NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
    redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master  1h
    redis-slave       10.0.21.92       <none>            6379/TCP      app-redis,role=slave   1h
    ...
    ```

    Result: The service is created with labels `app=redis` and `role=slave` to identify that the pods are running the Redis slaves.

Tip: It is helpful to set labels on your services themselves--as we've done here--to make it easy to locate them later.

The Redis master we created earlier is a single pod (REPLICAS = 1), while the Redis read slaves we are creating here are 'replicated' pods. In Kubernetes, a replication controller is responsible for managing the multiple instances of a replicated pod.

1. Use the file [redis-slave-controller.yaml](redis-slave-controller.yaml) to create the replication controller by running the `kubectl create -f` *`filename`* command:

    ```console
    $ kubectl create -f examples/guestbook-go/redis-slave-controller.yaml
    replicationcontroller "redis-slave" created
    ```

2. To verify that the redis-slave controller is running, run the `kubectl get rc` command:

    ```console
    $ kubectl get rc
    CONTROLLER              CONTAINER(S)            IMAGE(S)               SELECTOR                    REPLICAS
    redis-master            redis-master            gurpartap/redis        app=redis,role=master       1
    redis-slave             redis-slave             gurpartap/redis        app=redis,role=slave        2
    ...
    ```

    Result: The replication controller creates and configures the Redis slave pods through the redis-master service (name:port pair, in our example that's `redis-master:6379`).

    Example:
    The Redis slaves get started by the replication controller with the following command:

    ```console
    redis-server --slaveof redis-master 6379
    ```

3. To verify that the Redis master and slaves pods are running, run the `kubectl get pods` command:

    ```console
    $ kubectl get pods
    NAME                          READY     STATUS    RESTARTS   AGE
    redis-master-xx4uv            1/1       Running   0          18m
    redis-slave-b6wj4             1/1       Running   0          1m
    redis-slave-iai40             1/1       Running   0          1m
    ...
    ```

    Result: You see the single Redis master and two Redis slave pods.

### Step Three: Create the guestbook

Just like the others, we create a service to group the guestbook pods but this time, to make the guestbook front-end externally visible, we specify `"type": "LoadBalancer"`.

1. Use the [guestbook-service.yaml](guestbook-service.yaml) file to create the guestbook service by running the `kubectl create -f` *`filename`* command:

    ```console
    $ kubectl create -f examples/guestbook-go/guestbook-service.yaml
    ```

2. To verify that the guestbook service is up, list the services you created in the cluster with the `kubectl get services` command:

    ```console
    $ kubectl get services
    NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
    guestbook         10.0.217.218     146.148.81.8      3000/TCP      app=guestbook          1h
    redis-master      10.0.136.3       <none>            6379/TCP      app=redis,role=master  1h
    redis-slave       10.0.21.92       <none>            6379/TCP      app-redis,role=slave   1h
    ...
    ```

    Result: The service is created with label `app=guestbook`.

This is a simple Go `net/http` ([negroni](https://github.com/codegangsta/negroni) based) server that is configured to talk to either the slave or master services depending on whether the request is a read or a write. The pods we are creating expose a simple JSON interface and serves a jQuery-Ajax based UI. Like the Redis read slaves, these pods are also managed by a replication controller.

1. Use the [guestbook-controller.yaml](guestbook-controller.yaml) file to create the guestbook replication controller by running the `kubectl create -f` *`filename`* command:

    ```console
    $ kubectl create -f examples/guestbook-go/guestbook-controller.yaml
    replicationcontroller "guestbook" created
    ```

2. To verify that the guestbook replication controller is running, run the `kubectl get rc` command:

    ```console
    $ kubectl get rc
    CONTROLLER            CONTAINER(S)         IMAGE(S)                    SELECTOR                  REPLICAS
    guestbook             guestbook            kubernetes/guestbook:v2     app=guestbook             3
    redis-master          redis-master         gurpartap/redis             app=redis,role=master     1
    redis-slave           redis-slave          gurpartap/redis             app=redis,role=slave      2
    ...
    ```

3. To verify that the guestbook pods are running (it might take up to thirty seconds to create the pods), list the pods you created in cluster with the `kubectl get pods` command:

    ```console
    $ kubectl get pods
    NAME                           READY     STATUS    RESTARTS   AGE
    guestbook-3crgn                1/1       Running   0          2m
    guestbook-gv7i6                1/1       Running   0          2m
    guestbook-x405a                1/1       Running   0          2m
    redis-master-xx4uv             1/1       Running   0          23m
    redis-slave-b6wj4              1/1       Running   0          6m
    redis-slave-iai40              1/1       Running   0          6m
    ... 
    ```

    Result: You see a single Redis master, two Redis slaves, and three guestbook pods.

### Step Four: View the guestbook

You can now play with the guestbook that you just created by opening it in a browser (it might take a few moments for the guestbook to come up).

 * **Local Host:**
    If you are running Kubernetes locally, to view the guestbook, navigate to `http://localhost:3000` in your browser.

 * **Remote Host:**
    1. To view the guestbook on a remote host, locate the external IP of the load balancer in the **IP** column of the `kubectl get services` output. In our example, the internal IP address is `10.0.217.218` and the external IP address is `146.148.81.8` (*Note: you might need to scroll to see the IP column*).

    2. Append port `3000` to the IP address (for example `http://146.148.81.8:3000`), and then navigate to that address in your browser.

    Result: The guestbook displays in your browser:

    ![Guestbook](guestbook-page.png)

    **Further Reading:**
    If you're using Google Compute Engine, see the details about limiting traffic to specific sources at [Google Compute Engine firewall documentation][gce-firewall-docs].

[cloud-console]: https://console.developer.google.com
[gce-firewall-docs]: https://cloud.google.com/compute/docs/networking#firewalls

### Step Five: Cleanup

After you're done playing with the guestbook, you can cleanup by deleting the guestbook service and removing the associated resources that were created, including load balancers, forwarding rules, target pools, and Kubernetes replication controllers and services.

Delete all the resources by running the following `kubectl delete -f` *`filename`* command:

```console
$ kubectl delete -f examples/guestbook-go
replicationcontroller "guestbook" deleted
service "guestbook" deleted
replicationcontroller "redis-master" deleted
service "redis-master" deleted
replicationcontroller "redis-slave" deleted
service "redis-slave" deleted
```

Tip: To turn down your Kubernetes cluster, follow the corresponding instructions in the version of the
[Getting Started Guides](../../docs/getting-started-guides/) that you previously used to create your cluster.

### Troubleshooting

If you are having trouble bringing up your guestbook app, double check that your external IP is properly defined for your frontend service, and that the firewall for your cluster nodes is open to port 80.

Then, see the [troubleshooting documentation](../../docs/troubleshooting.md) for a further list of common issues and how you can diagnose them.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/guestbook-go/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
