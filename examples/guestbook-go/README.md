## GuestBook example

This example shows how to build a simple multi-tier web application using Kubernetes and Docker. It consists of a web frontend, a redis master for storage and a replicated set of redis slaves.

### Step Zero: Prerequisites

This example assumes that you have a working cluster (see the  [Getting Started Guides](../../docs/getting-started-guides)).
A Google Container Engine specific version of this tutoriual can be found at [https://cloud.google.com/container-engine/docs/tutorials/guestbook](https://cloud.google.com/container-engine/docs/tutorials/guestbook).

### Step One: Turn up the redis master.

Use the file `examples/guestbook-go/redis-master-controller.json` to create a [replication controller](../../docs/replication-controller.md) which manages a single [pod](../../docs/pods.md). The pod runs a redis key-value server in a container. Using a replication controller is the preferred way to launch long-running pods, even for 1 replica, so the pod will benefit from self-healing mechanism in Kubernetes.

Create the redis master replication controller in your Kubernetes cluster using the `kubectl` CLI and the file that specifies the replication controller [examples/guestbook-go/redis-master-controller.json](redis-master-controller.json):

```shell
$ kubectl create -f examples/guestbook-go/redis-master-controller.json
replicationcontrollers/redis-master
```

Once that's up you can list the replication controllers in the cluster:
```shell
$ kubectl get rc
CONTROLLER                     CONTAINER(S)            IMAGE(S)                                          SELECTOR                                   REPLICAS                                               
redis-master                   redis-master            gurpartap/redis                                   app=redis,role=master                      1
...

```

List pods in the cluster to verify the master is running. You'll see a single redis master pod and perhaps
some other system pods. The state of the pod and number of restarts and the duration it has been
executing for will also be reported (may take up to thirty seconds for the state to becoming ready and running).

```shell
$ kubectl get pods
NAME                                           READY     REASON    RESTARTS   AGE
redis-master-xx4uv                             1/1       Running   0          1m
...
```

If you ssh to that machine, you can run `docker ps` to see the actual pod:

```shell
me@workstation$ gcloud compute ssh --zone us-central1-b kubernetes-minion-bz1p

me@kubernetes-minion-3:~$ sudo docker ps
CONTAINER ID        IMAGE                                  COMMAND                CREATED             STATUS
d5c458dabe50        gurpartap/redis:latest                 "/usr/local/bin/redi   5 minutes ago       Up 5 minutes
```

(Note that initial `docker pull` may take a few minutes, depending on network conditions.)

### Step Two: Turn up the master service.
A Kubernetes '[service](../../docs/services.md)' is a named load balancer that proxies traffic to one or more containers. The services in a Kubernetes cluster are discoverable inside other containers via environment variables or DNS. Services find the containers to load balance based on pod labels.

The pod that you created in Step One has the label `app=redis` and `role=master`. The selector field of the service determines which pods will receive the traffic sent to the service.  Use the file [examples/guestbook-go/redis-master-service.json](redis-master-service.json) to create the service in the `kubectl` cli:

```shell
$ kubectl create -f examples/guestbook-go/redis-master-service.json
services/redis-master

$ kubectl get services
NAME                    LABELS                                                                                              SELECTOR                        IP(S)          PORT(S)
redis-master            app=redis,role=master                                                                               app=redis,role=master           10.0.136.3     6379/TCP
...
```

This will cause all new pods to see the redis master apparently running on `$REDIS_MASTER_SERVICE_HOST` at port 6379, or running on `redis-master:6379`. Once created, the service proxy on each node is configured to set up a proxy on the specified port (in this case port 6379).

### Step Three: Turn up the replicated slave pods.
Although the redis master is a single pod, the redis read slaves are a 'replicated' pod. In Kubernetes, a replication controller is responsible for managing multiple instances of a replicated pod.

Use the file [examples/guestbook-go/redis-slave-controller.json](redis-slave-controller.json) to create the replication controller:

```shell
$ kubectl create -f examples/guestbook-go/redis-slave-controller.json
replicationcontrollers/redis-slave

$ kubectl get rc
CONTROLLER                     CONTAINER(S)            IMAGE(S)                                          SELECTOR                                   REPLICAS                                               
redis-master                   redis-master            gurpartap/redis                                   app=redis,role=master                      1
redis-slave                    redis-slave             gurpartap/redis                                   app=redis,role=slave                       2
...
```

The redis slave configures itself by looking for the redis-master service name:port pair. In particular, the redis slave is started with the following command:

```shell
redis-server --slaveof redis-master 6379
```

Once that's up you can list the pods in the cluster, to verify that the master and slaves are running:

```shell
$ kubectl get pods
NAME                                           READY     REASON    RESTARTS   AGE
redis-master-xx4uv                             1/1       Running   0          18m
redis-slave-b6wj4                              1/1       Running   0          1m
redis-slave-iai40                              1/1       Running   0          1m
...

```

You will see a single redis master pod and two redis slave pods.

### Step Four: Create the redis slave service.

Just like the master, we want to have a service to proxy connections to the read slaves. In this case, in addition to discovery, the slave service provides transparent load balancing to clients. The service specification for the slaves
is in [examples/guestbook-go/redis-slave-service.json](redis-slave-service.json)

This time the selector for the service is `app=redis,role=slave`, because that identifies the pods running redis slaves. It may also be helpful to set labels on your service itself--as we've done here--to make it easy to locate them later.

Now that you have created the service specification, create it in your cluster with the `kubectl` CLI:

```shell
$ kubectl create -f examples/guestbook-go/redis-slave-service.json
services/redis-slave

$ kubectl get services
NAME                    LABELS                                                                                              SELECTOR                        IP(S)          PORT(S)
redis-master            app=redis,role=master                                                                               app=redis,role=master           10.0.136.3     6379/TCP
redis-slave             app=redis,role=slave                                                                                app=redis,role=slave            10.0.21.92     6379/TCP
...

```

### Step Five: Create the guestbook pod.

This is a simple Go net/http ([negroni](https://github.com/codegangsta/negroni) based) server that is configured to talk to either the slave or master services depending on whether the request is a read or a write. It exposes a simple JSON interface, and serves a jQuery-Ajax based UX. Like the redis read slaves it is a replicated service instantiated by a replication controller.

The pod is described in the file [examples/guestbook-go/guestbook-controller.json](guestbook-controller.json). Using this file, you can turn up your guestbook with:

```shell
$ kubectl create -f examples/guestbook-go/guestbook-controller.json
replicationcontrollers/guestbook

$ kubectl get replicationControllers
CONTROLLER                     CONTAINER(S)            IMAGE(S)                                          SELECTOR                                   REPLICAS
guestbook                      guestbook               kubernetes/guestbook:v2                           app=guestbook                              3
redis-master                   redis-master            gurpartap/redis                                   app=redis,role=master                      1
redis-slave                    redis-slave             gurpartap/redis                                   app=redis,role=slave                       2
...
```

Once that's up (it may take ten to thirty seconds to create the pods) you can list the pods in the cluster, to verify that the master, slaves and guestbook frontends are running:

```shell
$ kubectl get pods
NAME                                           READY     REASON    RESTARTS   AGE
guestbook-3crgn                                1/1       Running   0          2m
guestbook-gv7i6                                1/1       Running   0          2m
guestbook-x405a                                1/1       Running   0          2m
redis-master-xx4uv                             1/1       Running   0          23m
redis-slave-b6wj4                              1/1       Running   0          6m
redis-slave-iai40                              1/1       Running   0          6m
... 
```

You will see a single redis master pod, two redis slaves, and three guestbook pods.

### Step Six: Create the guestbook service.

Just like the others, you want a service to group your guestbook pods.  The service specification for the guestbook is in [examples/guestbook-go/guestbook-service.json](guestbook-service.json).  There's a twist this time - because we want it to be externally visible, we set `"type": "LoadBalancer"` for the service.

```shell
$ kubectl create -f examples/guestbook-go/guestbook-service.json

      An external load-balanced service was created.  On many platforms (e.g. Google Compute Engine),
      you will also need to explicitly open a Firewall rule for the service port(s) (tcp:3000) to serve traffic.

      See https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs/services-firewall.md for more details.

$ kubectl get services
NAME                    LABELS                                                                                              SELECTOR                        IP(S)          PORT(S)
guestbook               app=guestbook                                                                                       app=guestbook                   10.0.217.218   3000/TCP
                                                                                                                                                            146.148.81.8   
redis-master            app=redis,role=master                                                                               app=redis,role=master           10.0.136.3     6379/TCP
redis-slave             app=redis,role=slave                                                                                app=redis,role=slave            10.0.21.92     6379/TCP
...
```
To play with the service itself, find the external IP of the load balancer. This is reported in the IP column for the guestbook services which shows
an internal IP address 10.0.217.218 and an external IP address 146.148.81.8 (you may need to scroll right in the box
above to see the IP column. It make take a few moments to show up) after which you can
visit port 3000 of that IP address e.g. `http://146.148.81.8:3000`.

**NOTE:** You may need to open the firewall for port 3000 using the [console][cloud-console] or the `gcloud` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion`:

```shell
$ gcloud compute firewall-rules create --allow=tcp:3000 --target-tags=kubernetes-minion kubernetes-minion-3000
```
For Google Container Engine clusters the nodes are tagged differently. See the [Google Container Engine Guestbook example](https://cloud.google.com/container-engine/docs/tutorials/guestbook).

When you visit the external IP address of the guestbook service in a browser you should see something like this:

![Guestbook](guestbook-page.png)

If you are running Kubernetes locally, you can just visit http://localhost:3000
For details about limiting traffic to specific sources, see the [Google Compute Engine firewall documentation][gce-firewall-docs].

[cloud-console]: https://console.developer.google.com
[gce-firewall-docs]: https://cloud.google.com/compute/docs/networking#firewalls

### Step Seven: Cleanup

You should delete the service which will remove any associated resources that were created e.g. load balancers, forwarding rules and target pools. All the resources (replication controllers and service) can be deleted with a single command:
```shell
$ kubectl delete -f examples/guestbook-go
guestbook-controller
guestbook
redis-master-controller
redis-master
redis-slave-controller
redis-slave
```

To turn down your Kubernetes cluster follow the appropriate instructions in the
[Getting Started Guides](../../docs/getting-started-guides) for your type of cluster.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/guestbook-go/README.md?pixel)]()
