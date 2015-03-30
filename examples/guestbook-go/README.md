## GuestBook example

This example shows how to build a simple multi-tier web application using LMKTFY and Docker. It consists of a web frontend, a redis master for storage and a replicated set of redis slaves.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a LMKTFY cluster](../../docs/getting-started-guides):

```shell
$ cd lmktfy
$ hack/dev-build-and-up.sh
```

### Step One: Turn up the redis master.

Use the file `examples/guestbook-go/redis-master-controller.json` to create a replication controller which manages a single pod. The pod runs a redis key-value server in a container. Using a replication controller is the preferred way to launch long-running pods, even for 1 replica, so the pod will benefit from self-healing mechanism in lmktfy.

Create the redis master replication controller in your LMKTFY cluster using the `lmktfyctl` CLI:

```shell
$ cluster/lmktfyctl.sh create -f examples/guestbook-go/redis-master-controller.json
```

Once that's up you can list the replication controllers in the cluster:
```shell
$ cluster/lmktfyctl.sh get rc
CONTROLLER                             CONTAINER(S)            IMAGE(S)                            SELECTOR                     REPLICAS
redis-master-controller                redis-master            gurpartap/redis                     name=redis,role=master       1
```

List pods in cluster to verify the master is running. You'll see a single redis master pod. It will also display the machine that the pod is running on once it gets placed (may take up to thirty seconds).

```shell
$ cluster/lmktfyctl.sh get pods
POD                      IP                  CONTAINER(S)        IMAGE(S)            HOST                                                             LABELS                   STATUS
redis-master-pod-hh2gd   10.244.3.7          redis-master        gurpartap/redis     lmktfy-minion-4.c.lucid-walker-725.internal/104.154.49.134   name=redis,role=master   Running
```

If you ssh to that machine, you can run `docker ps` to see the actual pod:

```shell
me@workstation$ gcloud compute ssh --zone us-central1-b lmktfy-minion-4

me@lmktfy-minion-3:~$ sudo docker ps
CONTAINER ID        IMAGE                                  COMMAND                CREATED             STATUS
d5c458dabe50        gurpartap/redis:latest                 "/usr/local/bin/redi   5 minutes ago       Up 5 minutes
```

(Note that initial `docker pull` may take a few minutes, depending on network conditions.)

### Step Two: Turn up the master service.
A LMKTFY 'service' is a named load balancer that proxies traffic to one or more containers. The services in a LMKTFY cluster are discoverable inside other containers via environment variables or DNS. Services find the containers to load balance based on pod labels.

The pod that you created in Step One has the label `name=redis` and `role=master`. The selector field of the service determines which pods will receive the traffic sent to the service.  Use the file `examples/guestbook-go/redis-master-service.json` to create the service in the `lmktfyctl` cli:

```shell
$ cluster/lmktfyctl.sh create -f examples/guestbook-go/redis-master-service.json

$ cluster/lmktfyctl.sh get services
NAME                    LABELS                                    SELECTOR                     IP                  PORT
redis-master            <none>                                    name=redis,role=master       10.0.186.234        6379
```

This will cause all new pods to see the redis master apparently running on $REDIS_MASTER_SERVICE_HOST at port 6379, or running on 'redis-master:6379'. Once created, the service proxy on each node is configured to set up a proxy on the specified port (in this case port 6379).

### Step Three: Turn up the replicated slave pods.
Although the redis master is a single pod, the redis read slaves are a 'replicated' pod. In LMKTFY, a replication controller is responsible for managing multiple instances of a replicated pod.

Use the file `examples/guestbook-go/redis-slave-controller.json` to create the replication controller:

```shell
$ cluster/lmktfyctl.sh create -f examples/guestbook-go/redis-slave-controller.json

$ cluster/lmktfyctl.sh get rc
CONTROLLER                             CONTAINER(S)            IMAGE(S)                            SELECTOR                     REPLICAS
redis-master-controller                redis-master            gurpartap/redis                     name=redis,role=master       1
redis-slave-controller                 redis-slave             gurpartap/redis                     name=redis,role=slave        2
```

The redis slave configures itself by looking for the redis-master service name:port pair. In particular, the redis slave is started with the following command:

```shell
redis-server --slaveof redis-master 6379
```

Once that's up you can list the pods in the cluster, to verify that the master and slaves are running:

```shell
$ cluster/lmktfyctl.sh get pods
POD                                          IP                  CONTAINER(S)            IMAGE(S)                            HOST                                                             LABELS                                   STATUS
redis-master-pod-hh2gd                       10.244.3.7          redis-master            gurpartap/redis                     lmktfy-minion-4.c.lucid-walker-725.internal/104.154.49.134   name=redis,role=master                   Running
redis-slave-controller-i7hvs                 10.244.2.7          redis-slave             gurpartap/redis                     lmktfy-minion-3.c.lucid-walker-725.internal/104.154.52.39    name=redis,role=slave                    Running
redis-slave-controller-nyxxv                 10.244.1.6          redis-slave             gurpartap/redis                     lmktfy-minion-2.c.lucid-walker-725.internal/130.211.144.5    name=redis,role=slave                    Running
```

You will see a single redis master pod and two redis slave pods.

### Step Four: Create the redis slave service.

Just like the master, we want to have a service to proxy connections to the read slaves. In this case, in addition to discovery, the slave service provides transparent load balancing to clients. The service specification for the slaves is in `examples/guestbook-go/redis-slave-service.json`

This time the selector for the service is `name=redis,role=slave`, because that identifies the pods running redis slaves. It may also be helpful to set labels on your service itself--as we've done here--to make it easy to locate them later.

Now that you have created the service specification, create it in your cluster with the `lmktfyctl` CLI:

```shell
$ cluster/lmktfyctl.sh create -f examples/guestbook-go/redis-slave-service.json

$ cluster/lmktfyctl.sh get services
NAME                    LABELS                                    SELECTOR                     IP                  PORT
redis-master            <none>                                    name=redis,role=master       10.0.186.234        6379
redis-slave             name=redis,role=slave                     name=redis,role=slave        10.0.22.180         6379
```

### Step Five: Create the guestbook pod.

This is a simple Go net/http ([negroni](https://github.com/codegangsta/negroni) based) server that is configured to talk to either the slave or master services depending on whether the request is a read or a write. It exposes a simple JSON interface, and serves a jQuery-Ajax based UX. Like the redis read slaves it is a replicated service instantiated by a replication controller.

The pod is described in the file `examples/guestbook-go/guestbook-controller.json`. Using this file, you can turn up your guestbook with:

```shell
$ cluster/lmktfyctl.sh create -f examples/guestbook-go/guestbook-controller.json

$ cluster/lmktfyctl.sh get replicationControllers
CONTROLLER                             CONTAINER(S)            IMAGE(S)                            SELECTOR                     REPLICAS
guestbook-controller                   guestbook               lmktfy/guestbook:v2             name=guestbook               3
redis-master-controller                redis-master            gurpartap/redis                     name=redis,role=master       1
redis-slave-controller                 redis-slave             gurpartap/redis                     name=redis,role=slave        2
```

Once that's up (it may take ten to thirty seconds to create the pods) you can list the pods in the cluster, to verify that the master, slaves and guestbook frontends are running:

```shell
$ cluster/lmktfyctl.sh get pods
POD                                          IP                  CONTAINER(S)            IMAGE(S)                            HOST                                                             LABELS                                   STATUS
guestbook-controller-182tv                   10.244.2.8          guestbook               lmktfy/guestbook:v2             lmktfy-minion-3.c.lucid-walker-725.internal/104.154.52.39    name=guestbook                           Running
guestbook-controller-jzjpe                   10.244.0.7          guestbook               lmktfy/guestbook:v2             lmktfy-minion-1.c.lucid-walker-725.internal/104.154.37.86    name=guestbook                           Running
guestbook-controller-zwk1b                   10.244.3.8          guestbook               lmktfy/guestbook:v2             lmktfy-minion-4.c.lucid-walker-725.internal/104.154.49.134   name=guestbook                           Running
redis-master-pod-hh2gd                       10.244.3.7          redis-master            gurpartap/redis                     lmktfy-minion-4.c.lucid-walker-725.internal/104.154.49.134   name=redis,role=master                   Running
redis-slave-controller-i7hvs                 10.244.2.7          redis-slave             gurpartap/redis                     lmktfy-minion-3.c.lucid-walker-725.internal/104.154.52.39    name=redis,role=slave                    Running
redis-slave-controller-nyxxv                 10.244.1.6          redis-slave             gurpartap/redis                     lmktfy-minion-2.c.lucid-walker-725.internal/130.211.144.5    name=redis,role=slave                    Running
```

You will see a single redis master pod, two redis slaves, and three guestbook pods.

### Step Six: Create the guestbook service.

Just like the others, you want a service to group your guestbook pods.  The service specification for the guestbook is in `examples/guestbook-go/guestbook-service.json`.  There's a twist this time - because we want it to be externally visible, we set the `createExternalLoadBalancer` flag on the service.

```shell
$ cluster/lmktfyctl.sh create -f examples/guestbook-go/guestbook-service.json

$ cluster/lmktfyctl.sh get services
NAME                    LABELS                                    SELECTOR                     IP                  PORT
guestbook               <none>                                    name=guestbook               10.0.12.110         3000
redis-master            <none>                                    name=redis,role=master       10.0.186.234        6379
redis-slave             name=redis,role=slave                     name=redis,role=slave        10.0.22.180         6379
```

To play with the service itself, find the external IP of the load balancer:

```shell
$ cluster/lmktfyctl.sh get services guestbook -o template --template='{{index . "publicIPs"}}'
current-context: "lmktfy-satnam_lmktfy"
Running: cluster/../cluster/gce/../../_output/dockerized/bin/linux/amd64/lmktfyctl get services guestbook -o template --template={{index . "publicIPs"}}
[104.154.87.59]$

```
and then visit port 3000 of that IP address e.g. `http://104.154.87.59:3000`.

You may need to open the firewall for port 3000 using the [console][cloud-console] or the `gcloud` tool. The following command will allow traffic from any source to instances tagged `lmktfy-minion`:

```shell
$ gcloud compute firewall-rules create --allow=tcp:3000 --target-tags=lmktfy-minion lmktfy-minion-3000
```

If you are running LMKTFY locally, you can just visit http://localhost:3000
For details about limiting traffic to specific sources, see the [GCE firewall documentation][gce-firewall-docs].

[cloud-console]: https://console.developer.google.com
[gce-firewall-docs]: https://cloud.google.com/compute/docs/networking#firewalls

### Step Seven: Cleanup

You should delete the service which will remove any associated resources that were created e.g. load balancers, forwarding rules and target pools. All the resources (pods, replication controllers and service) can be deleted with a single command:
```shell
$ cluster/lmktfyctl.sh delete -f examples/guestbook-go
current-context: "lmktfy-satnam_lmktfy"
Running: cluster/../cluster/gce/../../_output/dockerized/bin/linux/amd64/lmktfyctl delete -f examples/guestbook-go
guestbook-controller
guestbook
redis-master-controller
redis-master
redis-slave-controller
redis-slave
```
To turn down a LMKTFY cluster:

```shell
$ cluster/lmktfy-down.sh
```
