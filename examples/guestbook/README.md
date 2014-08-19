## GuestBook example

This example shows how to build a simple multi-tier web application using Kubernetes and Docker.

The example combines a web frontend, a redis master for storage and a replicated set of redis slaves.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](https://github.com/GoogleCloudPlatform/kubernetes-new#contents):

### Step One: Turn up the redis master.

Use the file `examples/guestbook/redis-master.json` which describes a single pod running a redis key-value server in a container.

```javascript
{
  "id": "redis-master-2",
  "kind": "Pod",
  "apiVersion": "v1beta1",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "redis-master-2",
      "containers": [{
        "name": "master",
        "image": "dockerfile/redis",
        "ports": [{
          "containerPort": 6379,
          "hostPort": 6379
        }]
      }]
    }
  },
  "labels": {
    "name": "redis-master"
  }
}
```

Create the redis pod in your Kubernetes cluster using the `kubecfg` CLI:

```shell
$ cluster/kubecfg.sh -c examples/guestbook/redis-master.json create pods
```

Once that's up you can list the pods in the cluster, to verify that the master is running:

```shell
cluster/kubecfg.sh list pods
```

You'll see a single redis master pod. It will also display the machine that the pod is running on once it gets placed (may take up to thirty seconds).

```
Name                Image(s)            Host                                          Labels
----------          ----------          ----------                                    ----------
redis-master-2      dockerfile/redis    kubernetes-minion-3.c.briandpe-api.internal   name=redis-master
```

If you ssh to that machine, you can run `docker ps` to see the actual pod:

```shell
$ gcutil ssh --zone us-central1-b kubernetes-minion-3
$ sudo docker ps

me@kubernetes-minion-3:~$ sudo docker ps
CONTAINER ID  IMAGE  COMMAND  CREATED  STATUS  PORTS  NAMES
417ab993cdf8  dockerfile/redis:latest  redis-server /etc/re  8 minutes ago Up 8 minutes  0.0.0.0:6379->6379/tcp  master--redis_-_master_-_2--6b944b49
```

(Note that initial `docker pull` may take a few minutes, depending on network conditions.)

### Step Two: Turn up the master service.
A Kubernetes 'service' is a named load balancer that proxies traffic to one or more containers. The services in a Kubernetes cluster are discoverable inside other containers via environment variables. Services find the containers to load balance based on pod labels.

The pod that you created in Step One has the label `name=redis-master`. The selector field of the service determines which pods will receive the traffic sent to the service.  Use the file `examples/guestbook/redis-master-service.json`

```js
{
  "id": "redismaster",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 10000,
  "selector": {
    "name": "redis-master"
  }
}
```

to create the service with the `kubecfg` cli:

```shell
$ cluster/kubecfg.sh -c examples/guestbook/redis-master-service.json create services
Name                Labels              Selector            Port
----------          ----------          ----------          ----------
redismaster                             name=redis-master   10000
```

This will cause all pods to see the redis master apparently running on localhost:10000.

Once created, the service proxy on each minion is configured to set up a proxy on the specified port (in this case port 10000).

### Step Three: Turn up the replicated slave pods.
Although the redis master is a single pod, the redis read slaves are a 'replicated' pod. In Kubernetes, a replication controller is responsible for managing multiple instances of a replicated pod.

Use the file `examples/guestbook/redis-slave-controller.json`

```js
{ 
  "id": "redisSlaveController",
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "desiredState": {
    "replicas": 2,
    "replicaSelector": {"name": "redisslave"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "redisSlaveController",
           "containers": [{
             "name": "slave",
             "image": "brendanburns/redis-slave",
             "ports": [{"containerPort": 6379, "hostPort": 6380}]
           }]
         }
       },
       "labels": {"name": "redisslave"}
      }},
  "labels": {"name": "redisslave"}
}
```

to create the service by running:

```shell
$ cluster/kubecfg.sh -c examples/guestbook/redis-slave-controller.json create replicationControllers
Name                   Image(s)                   Selector            Replicas
----------             ----------                 ----------          ----------
redisSlaveController   brendanburns/redis-slave   name=redisslave     2
```

The redis slave configures itself by looking for the Kubernetes service environment variables in the container environment.  In particular, the redis slave is started with the following command:

```shell
redis-server --slaveof $SERVICE_HOST $REDISMASTER_SERVICE_PORT
```

Once that's up you can list the pods in the cluster, to verify that the master and slaves are running:

```shell
$ cluster/kubecfg.sh list pods
Name                Image(s)                   Host                                          Labels
----------          ----------                 ----------                                    ----------
redis-master-2      dockerfile/redis           kubernetes-minion-3.c.briandpe-api.internal   name=redis-master
4d65822107fcfd52    brendanburns/redis-slave   kubernetes-minion-3.c.briandpe-api.internal   name=redisslave,replicationController=redisSlaveController
78629a0f5f3f164f    brendanburns/redis-slave   kubernetes-minion-4.c.briandpe-api.internal   name=redisslave,replicationController=redisSlaveController
```

You will see a single redis master pod and two redis slave pods.

### Step Four: Create the redis slave service.

Just like the master, we want to have a service to proxy connections to the read slaves.  In this case, in addition to discovery, the slave service provides transparent load balancing to clients.  The service specification for the slaves is in `examples/guestbook/redis-slave-service.json`

```js
{
  "id": "redisslave",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 10001,
  "labels": {
    "name": "redisslave"
  },
  "selector": {
    "name": "redisslave"
  }
}
```

This time the selector for the service is `name=redisslave`, because that identifies the pods running redis slaves. It may also be helpful to set labels on your service itself--as we've done here--to make it easy to locate them with the `kubecfg -l "label=value" list services` command.

Now that you have created the service specification, create it in your cluster with the `kubecfg` CLI:

```shell
$ cluster/kubecfg.sh -c examples/guestbook/redis-slave-service.json create services
Name                Labels              Selector            Port
----------          ----------          ----------          ----------
redisslave          name=redisslave     name=redisslave     10001
```

### Step Five: Create the frontend pod.

This is a simple PHP server that is configured to talk to either the slave or master services depending on whether the request is a read or a write. It exposes a simple AJAX interface, and serves an angular-based UX. Like the redis read slaves it is a replicated service instantiated by a replication controller.

The pod is described in the file `examples/guestbook/frontend-controller.json`:

```js
{
  "id": "frontendController",
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "desiredState": {
    "replicas": 3,
    "replicaSelector": {"name": "frontend"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "frontendController",
           "containers": [{
             "name": "php-redis",
             "image": "brendanburns/php-redis",
             "ports": [{"containerPort": 80, "hostPort": 8000}]
           }]
         }
       },
       "labels": {"name": "frontend"}
      }},
  "labels": {"name": "frontend"}
}
```

Using this file, you can turn up your frontend with:

```shell
$ cluster/kubecfg.sh -c examples/guestbook/frontend-controller.json create replicationControllers
Name                 Image(s)                 Selector            Replicas
----------           ----------               ----------          ----------
frontendController   brendanburns/php-redis   name=frontend       3
```

Once that's up (it may take ten to thirty seconds to create the pods) you can list the pods in the cluster, to verify that the master, slaves and frontends are running:

```shell
$ cluster/kubecfg.sh list pods
Name                Image(s)                   Host                                          Labels
----------          ----------                 ----------                                    ----------
redis-master-2      dockerfile/redis           kubernetes-minion-3.c.briandpe-api.internal   name=redis-master
4d65822107fcfd52    brendanburns/redis-slave   kubernetes-minion-3.c.briandpe-api.internal   name=redisslave,replicationController=redisSlaveController
380704bb7b4d7c03    brendanburns/php-redis     kubernetes-minion-3.c.briandpe-api.internal   name=frontend,replicationController=frontendController
55104dc76695721d    brendanburns/php-redis     kubernetes-minion-2.c.briandpe-api.internal   name=frontend,replicationController=frontendController
365a858149c6e2d1    brendanburns/php-redis     kubernetes-minion-1.c.briandpe-api.internal   name=frontend,replicationController=frontendController
78629a0f5f3f164f    brendanburns/redis-slave   kubernetes-minion-4.c.briandpe-api.internal   name=redisslave,replicationController=redisSlaveController
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
      'host'   => getenv('SERVICE_HOST'),
      'port'   => getenv('REDISMASTER_SERVICE_PORT'),
    ]);
    $client->set($_GET['key'], $_GET['value']);
    print('{"message": "Updated"}');
  } else {
    $read_port = getenv('REDISMASTER_SERVICE_PORT');

    if (isset($_ENV['REDISSLAVE_SERVICE_PORT'])) {
      $read_port = getenv('REDISSLAVE_SERVICE_PORT');
    }
    $client = new Predis\Client([
      'scheme' => 'tcp',
      'host'   => getenv('SERVICE_HOST'),
      'port'   => $read_port,
    ]);

    $value = $client->get($_GET['key']);
    print('{"data": "' . $value . '"}');
  }
} else {
  phpinfo();
} ?>
```

To play with the service itself, find the name of a frontend, grab the external IP of that host from the [Google Cloud Console][cloud-console] or the `gcutil` tool, and visit `http://<host-ip>:8000`. 

```shell
$ gcutil listinstances
```

You may need to open the firewall for port 8000 using the [console][cloud-console] or the `gcutil` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion`:

```shell
$ gcutil addfirewall --allowed=tcp:8000 --target_tags=kubernetes-minion kubernetes-minion-8000
```

If you are running Kubernetes locally, you can just visit http://localhost:8000
For details about limiting traffic to specific sources, see the [gcutil documentation][gcutil-docs]

[cloud-console]: https://console.developer.google.com
[gcutil-docs]: https://developers.google.com/compute/docs/gcutil/reference/firewall#addfirewall
