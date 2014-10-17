## GuestBook example

This example shows how to build a simple multi-tier web application using Kubernetes and Docker.

The example combines a web frontend, a redis master for storage and a replicated set of redis slaves.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](https://github.com/GoogleCloudPlatform/kubernetes#contents):

```shell
$ cd kubernetes
$ hack/dev-build-and-up.sh
```

### Step One: Turn up the redis master.

Use the file `examples/guestbook-go/redis-master-pod.json` which describes a single pod running a redis key-value server in a container.

Create the redis pod in your Kubernetes cluster using the `kubectl` CLI:

```shell
$ cluster/kubectl.sh create -f examples/guestbook-go/redis-master-pod.json
```

Once that's up you can list the pods in the cluster, to verify that the master is running:

```shell
$ cluster/kubectl.sh get pods
```

You'll see a single redis master pod. It will also display the machine that the pod is running on once it gets placed (may take up to thirty seconds).

```
ID                  IMAGE(S)            HOST                                                      LABELS                     STATUS
redis-master-pod    gurpartap/redis     kubernetes-minion-3.c.thockin-dev.internal/86.75.30.9     name=redis,role=master     Waiting
```

If you ssh to that machine, you can run `docker ps` to see the actual pod:

```shell
$ gcutil ssh --zone us-central1-b kubernetes-minion-3
$ sudo docker ps

me@kubernetes-minion-3:~$ sudo docker ps
CONTAINER ID  IMAGE                   COMMAND              CREATED         STATUS
e443647cd064  gurpartap/redis:latest  redis-server /etc/r  2 minutes ago   Up 2 minutes
```

(Note that initial `docker pull` may take a few minutes, depending on network conditions.)

### Step Two: Turn up the master service.
A Kubernetes 'service' is a named load balancer that proxies traffic to one or more containers. The services in a Kubernetes cluster are discoverable inside other containers via environment variables. Services find the containers to load balance based on pod labels.

The pod that you created in Step One has the label `name=redis` and `role=master`. The selector field of the service determines which pods will receive the traffic sent to the service.  Use the file `examples/guestbook-go/redis-master-service.json`

To create the service with the `kubectl` cli:

```shell
$ cluster/kubectl.sh create -f examples/guestbook-go/redis-master-service.json

$ cluster/kubectl.sh get services
ID                  LABELS                     SELECTOR                   PORT
redis-master                                   name=redis,role=master     6379
```

This will cause all new pods to see the redis master apparently running on $REDIS_MASTER_SERVICE_HOST at port 6379.

Once created, the service proxy on each minion is configured to set up a proxy on the specified port (in this case port 6379).

### Step Three: Turn up the replicated slave pods.
Although the redis master is a single pod, the redis read slaves are a 'replicated' pod. In Kubernetes, a replication controller is responsible for managing multiple instances of a replicated pod.

Use the file `examples/guestbook-go/redis-slave-controller.json`

to create the replication controller by running:

```shell
$ cluster/kubectl.sh create -f examples/guestbook-go/redis-slave-controller.json

$ cluster/kubectl.sh get replicationControllers
ID                       IMAGE(S)                    SELECTOR                   REPLICAS
redis-slave-controller   gurpartap/redis             name=redis,role=slave      2
```

The redis slave configures itself by looking for the Kubernetes service environment variables in the container environment. In particular, the redis slave is started with the following command:

```shell
redis-server --slaveof $REDIS_MASTER_SERVICE_HOST $REDIS_MASTER_SERVICE_PORT
```

Once that's up you can list the pods in the cluster, to verify that the master and slaves are running:

```shell
$ cluster/kubectl.sh get pods
ID                                    IMAGE(S)            HOST                                                      LABELS                                                               STATUS
redis-master-pod                      gurpartap/redis     kubernetes-minion-1.c.thockin-dev.internal/86.75.30.9     name=redis,role=master                                               Running
1472fd26-54d6-11e4-90fd-42010af00690  gurpartap/redis     kubernetes-minion-4.c.thockin-dev.internal/12.34.56.78    name=redis,replicationController=redis-slave-controller,role=slave   Running
1473363e-54d6-11e4-90fd-42010af00690  gurpartap/redis     kubernetes-minion-3.c.thockin-dev.internal/9.3.19.76      name=redis,replicationController=redis-slave-controller,role=slave   Running
```

You will see a single redis master pod and two redis slave pods.

### Step Four: Create the redis slave service.

Just like the master, we want to have a service to proxy connections to the read slaves. In this case, in addition to discovery, the slave service provides transparent load balancing to clients. The service specification for the slaves is in `examples/guestbook-go/redis-slave-service.json`

This time the selector for the service is `name=redis,role=slave`, because that identifies the pods running redis slaves. It may also be helpful to set labels on your service itself--as we've done here--to make it easy to locate them later.

Now that you have created the service specification, create it in your cluster with the `kubectl` CLI:

```shell
$ cluster/kubectl.sh create -f examples/guestbook-go/redis-slave-service.json

$ cluster/kubectl.sh get services
ID                  LABELS                     SELECTOR                   PORT
redis-master                                   name=redis,role=master     6379
redis-slave         name=redis,role=slave      name=redis,role=slave      6379
```

### Step Five: Create the guestbook pod.

This is a simple Go net/http ([negroni](https://github.com/codegangsta/negroni) based) server that is configured to talk to either the slave or master services depending on whether the request is a read or a write. It exposes a simple JSON interface, and serves a jQuery-Ajax based UX. Like the redis read slaves it is a replicated service instantiated by a replication controller.

The pod is described in the file `examples/guestbook-go/guestbook-controller.json`:

Using this file, you can turn up your guestbook with:

```shell
$ cluster/kubectl.sh create -f examples/guestbook-go/guestbook-controller.json

$ cluster/kubectl.sh get replicationControllers
ID                       IMAGE(S)               SELECTOR                REPLICAS
redis-slave-controller   gurpartap/redis        name=redis,role=slave   2
guestbook-controller     kubernetes/guestbook   name=guestbook          3
```

Once that's up (it may take ten to thirty seconds to create the pods) you can list the pods in the cluster, to verify that the master, slaves and guestbook frontends are running:

```shell
$ cluster/kubectl.sh get pods
ID                                    IMAGE(S)              HOST                                                     LABELS                                                               STATUS
redis-master-pod                      gurpartap/redis       kubernetes-minion-1.c.thockin-dev.internal/86.75.30.9    name=redis,role=master                                               Running
1472fd26-54d6-11e4-90fd-42010af00690  gurpartap/redis       kubernetes-minion-4.c.thockin-dev.internal/12.34.56.78   name=redis,replicationController=redis-slave-controller,role=slave   Running
1473363e-54d6-11e4-90fd-42010af00690  gurpartap/redis       kubernetes-minion-3.c.thockin-dev.internal/9.3.19.76     name=redis,replicationController=redis-slave-controller,role=slave   Running
fc58aa01-54d6-11e4-90fd-42010af00690  kubernetes/guestbook  kubernetes-minion-1.c.thockin-dev.internal/1.18.19.78    name=guestbook,replicationController=guestbook-controller            Running
fc592fbb-54d6-11e4-90fd-42010af00690   kubernetes/guestbook   kubernetes-minion-1.c.thockin-dev.internal/12.9.20.9   name=guestbook,replicationController=guestbook-controller            Running
fc59569e-54d6-11e4-90fd-42010af00690   kubernetes/guestbook   kubernetes-minion-2.c.thockin-dev.internal/1.11.20.13  name=guestbook,replicationController=guestbook-controller            Running

```

You will see a single redis master pod, two redis slaves, and three guestbook pods.

### Step Six: Create the guestbook service.

Just like the others, you want a service to group your guestbook pods.  The service specification for the guestbook is in `examples/guestbook-go/guestbook-service.json`.  There's a twist this time - because we want it to be externally visible, we set the `createExternalLoadBalancer` flag on the service.

```shell
$ cluster/kubectl.sh create -f examples/guestbook-go/guestbook-service.json

$ cluster/kubectl.sh get services
ID                  LABELS                  SELECTOR                 IP                  PORT
redis-master                                name=redis,role=master   10.0.0.1            6379
redis-slave         name=redis,role=slave   name=redis,role=slave    10.0.0.2            6379
guestbook                                   name=guestbook           10.0.0.3            3000
```

To play with the service itself, find the external IP of the load balancer from the [Google Cloud Console][cloud-console] or the `gcutil` tool, and visit `http://<ip>:3000`.

```shell
$ gcutil getforwardingrule guestbook
+---------------+-----------------------------------+
| name          | guestbook                         |
| description   |                                   |
| creation-time | 2014-10-15T19:07:24.837-07:00     |
| region        | us-central1                       |
| ip            | 12.34.56.78                       |
| protocol      | TCP                               |
| port-range    | 3000-3000                         |
| target        | us-central1/targetPools/guestbook |
+---------------+-----------------------------------+
```

You may need to open the firewall for port 3000 using the [console][cloud-console] or the `gcutil` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion`:

```shell
$ gcutil addfirewall --allowed=tcp:3000 --target_tags=kubernetes-minion kubernetes-minion-3000
```

If you are running Kubernetes locally, you can just visit http://localhost:3000
For details about limiting traffic to specific sources, see the [gcutil documentation][gcutil-docs]

[cloud-console]: https://console.developer.google.com
[gcutil-docs]: https://developers.google.com/compute/docs/gcutil/reference/firewall#addfirewall

### Step Seven: Cleanup

To turn down a Kubernetes cluster:

```shell
$ cluster/kube-down.sh
```
