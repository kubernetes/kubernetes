## GuestBook example

This example shows how to build a simple, multi-tier web application using Kubernetes and Docker.

The example consists of:
- A web frontend
- A redis master (for storage and a replicated set of redis slaves)

The web front end interacts with the redis master via javascript redis API calls.

### Step Zero: Prerequisites

This example assumes that you have a basic understanding of kubernetes services and that you have forked the repository and [turned up a Kubernetes cluster](https://github.com/GoogleCloudPlatform/kubernetes#contents):
This example requires a kubernetes cluster.  

See the companion [Setup Kubernetes](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/guestbook/SETUP.md) for some quick notes on how to get started.

*If* you are running from source, replace commands such as "kubectl" below with calls to cluster/kubectl.sh.

### Step One: Fire up the redis master

Note: This redis-master is *not* highly available.  Making it highly available would be a very interesting, but intricate exersize - redis doesn't actually support multi-master deployments at the time of this writing, so high availability would be a somewhat tricky thing implement, and might involve periodic serialization to disk, and so on.

Use (or just create) the file `examples/guestbook/redis-master.json` which describes a single pod running a redis key-value server in a container:

```js
{
  "id": "redis-master",
  "kind": "Pod",
  "apiVersion": "v1beta1", 
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "redis-master",
      "containers": [{
        "name": "master",
        "image": "dockerfile/redis",
        "cpu": 100,
        "ports": [{
          "containerPort": 6379,   # containerPort: Where traffic to redis ultimately is routed to.
        }]
      }]
    }
  },
  "labels": {
    "name": "redis-master" # This label needed for when we start our redis-master service.
  }
}
```

Now, create the redis pod in your Kubernetes cluster by running:

```shell
kubectl create -f examples/guestbook/redis-master.json
```

Once that's up you can list the pods in the cluster, to verify that the master is running:

```shell
kubectl get pods
```

You'll see all kubernetes components, most importantly the redis master pod. It will also display the machine that the pod is running on once it gets placed (may take up to thirty seconds):

```shell
NAME                IMAGE(S)            HOST                                                       LABELS              STATUS
redis-master        dockerfile/redis    kubernetes-minion-2.c.myproject.internal/130.211.156.189   name=redis-master   Running

```

If you ssh to that machine, you can run `docker ps` to see the actual pod:

```shell
me@workstation$ gcloud compute ssh --zone us-central1-b kubernetes-minion-2

me@kubernetes-minion-2:~$ sudo docker ps
CONTAINER ID  IMAGE                     COMMAND                CREATED         STATUS        PORTS     NAMES
e3eed3e5e6d1  dockerfile/redis:latest   "redis-server /etc/re  2 minutes ago   Up 2 minutes            k8s_master.9c0a9146_redis-master.etcd_6296f4bd-70fa-11e4-8469-0800279696e1_45331ebc
```

(Note that initial `docker pull` may take a few minutes, depending on network conditions.  You can monitor the status of this by running `journalctl -f -u docker` to check when the image is being downloaded.  Of course, you can also run `journalctl -f -u kubelet` to see what state the kubelet is in as well during this time.

### Step Two: Fire up the master service
A Kubernetes 'service' is a named load balancer that proxies traffic to *one or more* containers. This is done using the *labels* metadata which we defined in the redis-master pod above.  As mentioned, in redis there is only one master, but we nevertheless still want to create a service for it.  Why?  Because it gives us a deterministic way to route to the single master using an elastic IP.  

The services in a Kubernetes cluster are discoverable inside other containers via environment variables. 

Services find the containers to load balance based on pod labels.

The pod that you created in Step One has the label `name=redis-master`. The selector field of the service determines *which pods will receive the traffic* sent to the service, and the port and containerPort information defines what port the service proxy will run at. 

Use the file `examples/guestbook/redis-master-service.json`:

```js
{
  "id": "redis-master",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 6379,
  "containerPort": 6379,
  "selector": {
    "name": "redis-master"
  },
  "labels": {
    "name": "redis-master"
  }
}
```

to create the service by running:

```shell
$ kubectl create -f examples/guestbook/redis-master-service.json
redis-master

$ cluster/kubectl.sh get services
NAME                LABELS              SELECTOR                                  IP                  PORT
kubernetes          <none>              component=apiserver,provider=kubernetes   10.0.29.11          443
kubernetes-ro       <none>              component=apiserver,provider=kubernetes   10.0.141.25         80
redis-master        name=redis-master   name=redis-master                         10.0.16.143         6379
```


This will cause all pods to see the redis master apparently running on <ip>:6379.  The traffic flow from slaves to masters can be described in two steps, like so.

- A *redis slave* will connect to "port" on the *redis master service*
- Traffic will be forwarded from the service "port" (on the service node) to the  *containerPort* on the pod which (a node the service listens to). 

Thus, once created, the service proxy on each minion is configured to set up a proxy on the specified port (in this case port 6379).

### Step Three: Fire up the replicated slave pods
Although the redis master is a single pod, the redis read slaves are a 'replicated' pod. In Kubernetes, a replication controller is responsible for managing multiple instances of a replicated pod.  The replicationController will automatically launch new Pods if the number of replicas falls (this is quite easy - and fun - to test, just kill the docker processes for your pods at will and watch them come back online on a new node shortly thereafter).

Use the file `examples/guestbook/redis-slave-controller.json`, which looks like this:

```js
{
  "id": "redis-slave-controller",
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "desiredState": {
    "replicas": 2,
    "replicaSelector": {"name": "redisslave"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "redis-slave-controller",
           "containers": [{
             "name": "slave",
             "image": "brendanburns/redis-slave",
             "cpu": 200,
             "ports": [{"containerPort": 6379, "hostPort": 6380}]
           }]
         }
       },
       "labels": {
         "name": "redisslave",
         "uses": "redis-master"
       }
      }},
  "labels": {"name": "redisslave"}
}
```

to create the replication controller by running:

```shell
$ kubectl create -f examples/guestbook/redis-slave-controller.json
redis-slave-controller

# kubectl.sh get replicationcontrollers
NAME                   IMAGE(S)                   SELECTOR            REPLICAS
redis-slave-controller brendanburns/redis-slave   name=redisslave     2
```

The redis slave configures itself by looking for the Kubernetes service environment variables in the container environment. In particular, the redis slave is started with the following command:

```shell
redis-server --slaveof ${REDIS_MASTER_SERVICE_HOST:-$SERVICE_HOST} $REDIS_MASTER_SERVICE_PORT
```

You might be curious about where the *REDIS_MASTER_SERVICE_HOST* is coming from.  It is provided to this container when it is launched via the kubernetes services, which create environment variables (there is a well defined syntax for how service names get transformed to environment variable names in the documentation linked above).

Once that's up you can list the pods in the cluster, to verify that the master and slaves are running:

```shell
$ kubectl get pods
NAME                                   IMAGE(S)                   HOST                                                        LABELS                              STATUS
redis-master                           dockerfile/redis           kubernetes-minion-2.c.myproject.internal/130.211.156.189    name=redis-master                   Running
ee68394b-7fca-11e4-a220-42010af0a5f1   brendanburns/redis-slave   kubernetes-minion-3.c.myproject.internal/130.211.179.212    name=redisslave,uses=redis-master   Running
ee694768-7fca-11e4-a220-42010af0a5f1   brendanburns/redis-slave   kubernetes-minion-4.c.myproject.internal/130.211.168.210    name=redisslave,uses=redis-master   Running
```

You will see a single redis master pod and two redis slave pods.

### Step Four: Create the redis slave service

Just like the master, we want to have a service to proxy connections to the read slaves. In this case, in addition to discovery, the slave service provides transparent load balancing to web app  clients. 

The service specification for the slaves is in `examples/guestbook/redis-slave-service.json`:

```js
{
  "id": "redisslave",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 6379,
  "containerPort": 6379,
  "labels": {
    "name": "redisslave"
  },
  "selector": {
    "name": "redisslave"
  }
}
```

This time the selector for the service is `name=redisslave`, because that identifies the pods running redis slaves. It may also be helpful to set labels on your service itself as we've done here to make it easy to locate them with the `cluster/kubectl.sh get services -l "label=value"` command.

Now that you have created the service specification, create it in your cluster by running:

```shell
$ cluster/kubectl.sh create -f examples/guestbook/redis-slave-service.json
redisslave

$ cluster/kubectl.sh get services

NAME                LABELS              SELECTOR                                  IP                  PORT
kubernetes          <none>              component=apiserver,provider=kubernetes   10.0.29.11          443
kubernetes-ro       <none>              component=apiserver,provider=kubernetes   10.0.141.25         80
redis-master        name=redis-master   name=redis-master                         10.0.16.143         6379
redisslave          name=redisslave     name=redisslave                           10.0.217.148        6379
```

### Step Five: Create the frontend pod

This is a simple PHP server that is configured to talk to either the slave or master services depending on whether the request is a read or a write. It exposes a simple AJAX interface, and serves an angular-based UX. Like the redis read slaves it is a replicated service instantiated by a replication controller.

It can now leverage writes to the load balancing redis-slaves, which can be highly replicated.

The pod is described in the file `examples/guestbook/frontend-controller.json`:

```js
{
  "id": "frontend-controller",
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "desiredState": {
    "replicas": 3,
    "replicaSelector": {"name": "frontend"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "frontend-controller",
           "containers": [{
             "name": "php-redis",
             "image": "kubernetes/example-guestbook-php-redis",
             "cpu": 100,
             "memory": 50000000,
             "ports": [{"containerPort": 80, "hostPort": 8000}]
           }]
         }
       },
       "labels": {
         "name": "frontend",
         "uses": "redisslave,redis-master"
       }
      }},
  "labels": {"name": "frontend"}
}
```

Using this file, you can turn up your frontend with:

```shell
$ cluster/kubectl.sh create -f examples/guestbook/frontend-controller.json
frontend-controller

$ cluster/kubectl.sh get replicationcontrollers
NAME                    IMAGE(S)                                   SELECTOR            REPLICAS
redis-slave-controller  brendanburns/redis-slave                   name=redisslave     2
frontend-controller     kubernetes/example-guestbook-php-redis     name=frontend       3
```

Once that's up (it may take ten to thirty seconds to create the pods) you can list the pods in the cluster, to verify that the master, slaves and frontends are running:

```shell
$ cluster/kubectl.sh get pods
NAME                                   IMAGE(S)                                   HOST                                                       LABELS                                       STATUS
redis-master                           dockerfile/redis                           kubernetes-minion-2.c.myproject.internal/130.211.156.189   name=redis-master                            Running
ee68394b-7fca-11e4-a220-42010af0a5f1   brendanburns/redis-slave                   kubernetes-minion-3.c.myproject.internal/130.211.179.212   name=redisslave,uses=redis-master            Running
ee694768-7fca-11e4-a220-42010af0a5f1   brendanburns/redis-slave                   kubernetes-minion-4.c.myproject.internal/130.211.168.210   name=redisslave,uses=redis-master            Running
9fbad0d6-7fcb-11e4-a220-42010af0a5f1   kubernetes/example-guestbook-php-redis     kubernetes-minion-1.c.myproject.internal/130.211.185.78    name=frontend,uses=redisslave,redis-master   Running
9fbbf70e-7fcb-11e4-a220-42010af0a5f1   kubernetes/example-guestbook-php-redis     kubernetes-minion-2.c.myproject.internal/130.211.156.189   name=frontend,uses=redisslave,redis-master   Running
9fbdbeca-7fcb-11e4-a220-42010af0a5f1   kubernetes/example-guestbook-php-redis     kubernetes-minion-4.c.myproject.internal/130.211.168.210   name=frontend,uses=redisslave,redis-master   Running
```

You will see a single redis master pod, two redis slaves, and three frontend pods.

The code for the PHP service looks like this:

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
      'host'   => getenv('REDIS_MASTER_SERVICE_HOST') ?: getenv('SERVICE_HOST'),
      'port'   => getenv('REDIS_MASTER_SERVICE_PORT'),
    ]);
    $client->set($_GET['key'], $_GET['value']);
    print('{"message": "Updated"}');
  } else {
    $read_port = getenv('REDIS_MASTER_SERVICE_PORT');

    if (isset($_ENV['REDISSLAVE_SERVICE_PORT'])) {
      $read_port = getenv('REDISSLAVE_SERVICE_PORT');
    }
    $client = new Predis\Client([
      'scheme' => 'tcp',
      'host'   => getenv('REDIS_MASTER_SERVICE_HOST') ?: getenv('SERVICE_HOST'),
      'port'   => $read_port,
    ]);

    $value = $client->get($_GET['key']);
    print('{"data": "' . $value . '"}');
  }
} else {
  phpinfo();
} ?>
```

To play with the service itself, find the name of a frontend,

### A few Google Container Engine specifics for playing around with the services.

- In GCE, you can grab the external IP of that host from the [Google Cloud Console][cloud-console] or the `gcloud` tool, and visit `http://<host-ip>:8000`.
```shell
$ gcloud compute instances list
```
In GCE, you also may need to open the firewall for port 8000 using the [console][cloud-console] or the `gcloud` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion`:

```shell
$ gcloud compute firewall-rules create --allow=tcp:8000 --target-tags=kubernetes-minion kubernetes-minion-8000
```

For GCE details about limiting traffic to specific sources, see the [GCE firewall documentation][gce-firewall-docs].

[cloud-console]: https://console.developer.google.com
[gce-firewall-docs]: https://cloud.google.com/compute/docs/networking#firewalls

In other environments, you can get the service IP from looking at the output of `kubectl get pods,services`, and modify your firewall using standard tools and services (firewalld, iptables, selinux) which you are already familar with.

And of course, finally, if you are running Kubernetes locally, you can just visit http://localhost:8000.  

### Step Six: Cleanup

To turn down a Kubernetes cluster, if you ran this from source, you can use

```shell
$ cluster/kube-down.sh
```

If you are in a live kubernetes cluster, you can just kill the pods, using a script such as this (obviously, read through it and make sure you understand it before running it blindly, as it will kill several pods automatically for you).

```shell
### First, kill services and controllers.
kubectl stop rc examples/guestbook/redis-slave-controller.json
kubectl stop rc examples/guestbook/frontend-controller.json
kubectl delete -f examples/guestbook/redis-master-service.json
kubectl delete -f examples/guestbook/redis-slave-service.json
kubectl delete pod redis-master # This is the only pod that requires manual removal.
```

