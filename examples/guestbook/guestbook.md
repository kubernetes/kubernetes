## GuestBook example

This example shows how to build a simple multi-tier web application using Kubernetes and Docker.

The example combines a web frontend, a redis master for storage and a replicated set of redis slaves.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](https://github.com/GoogleCloudPlatform/kubernetes-new#setup):

    $ cd kubernetes
    $ hack/dev-build-and-up.sh

### Step One: Turn up the redis master.

Create a file named `redis-master.json` describing a single pod, which runs a redis key-value server in a container.

```javascript
{
  "id": "redis-master-2",
  "desiredState": {
    "manifest": {
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

Once you have that pod file, you can create the redis pod in your Kubernetes cluster using the `cloudcfg` CLI:

```shell
$ cluster/cloudcfg.sh -c examples/guestbook/redis-master.json create /pods
```

Once that's up you can list the pods in the cluster, to verify that the master is running:

```shell
cluster/cloudcfg.sh list /pods
```

You'll see a single redis master pod. It will also display the machine that the pod is running on.

```javascript
Using master: kubernetes-master (external IP: 1.2.3.4)
{
    "kind": "cluster#podList",
    "items": [
        {
            "id": "redis-master-2",
            "labels": {
                "name": "redis-master"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "name": "master",
                            "image": "dockerfile/redis",
                            "ports": [
                                {
                                    "hostPort": 6379,
                                    "containerPort": 6379
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-3.c.{project-id}.internal"
            }
        }
    ]
}
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

The pod that you created in Step One has the label `name=redis-master`, so the corresponding service is defined by that label.  Create a file named `redis-master-service.json` that contains:

```js
{
  "id": "redismaster",
  "port": 10000,
  "labels": {
    "name": "redis-master"
  }
}
```

Once you have that service description, you can create the service with the `cloudcfg` cli:

```js
$ cluster/cloudcfg.sh -c examples/guestbook/redis-master-service.json create /services

Using master: kubernetes-master (external IP: 1.2.3.4)
{
    "id": "redismaster",
    "port": 10000,
    "labels": {
        "name": "redis-master"
    }
}

```

Once created, the service proxy on each minion is configured to set up a proxy on the specified port (in this case port 10000).

### Step Three: Turn up the replicated slave pods.
Although the redis master is a single pod, the redis read slaves are a 'replicated' pod. In Kubernetes, a replication controller is responsible for managing multiple instances of a replicated pod.

Create a file named `redis-slave-controller.json` that contains:

```js
  {
    "id": "redisSlaveController",
    "desiredState": {
      "replicas": 2,
      "replicasInSet": {"name": "redis-slave"},
      "podTemplate": {
        "desiredState": {
           "manifest": {
             "containers": [{
               "image": "brendanburns/redis-slave",
               "ports": [{"containerPort": 6379, "hostPort": 6380}]
             }]
           }
         },
         "labels": {"name": "redis-slave"}
        }},
    "labels": {"name": "redis-slave"}
  }
```

Then you can create the service by running:

```js
$ cluster/cloudcfg.sh -c examples/guestbook/redis-slave-controller.json create /replicationControllers

Using master: kubernetes-master (external IP: 1.2.3.4)
{
    "id": "redisSlaveController",
    "desiredState": {
        "replicas": 2,
        "replicasInSet": {
            "name": "redisslave"
        },
        "podTemplate": {
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "image": "brendanburns/redis-slave",
                            "ports": [
                                {
                                    "hostPort": 6380,
                                    "containerPort": 6379
                                }
                            ]
                        }
                    ]
                }
            },
            "labels": {
                "name": "redisslave"
            }
        }
    },
    "labels": {
        "name": "redisslave"
    }
}

```

The redis slave configures itself by looking for the Kubernetes service environment variables in the container environment.  In particular, the redis slave is started with the following command:

```shell
redis-server --slaveof $SERVICE_HOST $REDISMASTER_SERVICE_PORT
```

Once that's up you can list the pods in the cluster, to verify that the master and slaves are running:

```js
$ cluster/cloudcfg.sh list /pods

Using master: kubernetes-master (external IP: 23.236.49.160)
{
    "kind": "cluster#podList",
    "items": [
        {
            "id": "redis-master-2",
            "labels": {
                "name": "redis-master"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "name": "master",
                            "image": "dockerfile/redis",
                            "ports": [
                                {
                                    "hostPort": 6379,
                                    "containerPort": 6379
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-3.c.{project-id}.internal"
            }
        },
        {
            "id": "4d65822107fcfd52",
            "labels": {
                "name": "redisslave",
                "replicationController": "redisSlaveController"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "image": "brendanburns/redis-slave",
                            "ports": [
                                {
                                    "hostPort": 6380,
                                    "containerPort": 6379
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-3.c.{project-id}.internal"
            }
        },
        {
            "id": "78629a0f5f3f164f",
            "labels": {
                "name": "redisslave",
                "replicationController": "redisSlaveController"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "image": "brendanburns/redis-slave",
                            "ports": [
                                {
                                    "hostPort": 6380,
                                    "containerPort": 6379
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-2.c.{project-id}.internal"
            }
        }
    ]
}
```

You will see a single redis master pod and two redis slave pods.

### Step Four: Create the redis slave service.

Just like the master, we want to have a service to proxy connections to the read slaves.  In this case, in addition to discovery, the slave service provides transparent load balancing to clients.  As before, create a service specification:

```js
{
  "id": "redisslave",
  "port": 10001,
  "labels": {
    "name": "redis-slave"
  }
}
```

This time the label query for the service is `name=redis-slave`.

Now that you have created the service specification, create it in your cluster with the `cloudcfg` CLI:

```js
$ cluster/cloudcfg.sh -c examples/guestbook/redis-slave-service.json create /services

Using master: kubernetes-master (external IP: 1.2.3.4)
{
    "id": "redisslave",
    "port": 10001,
    "labels": {
        "name": "redisslave"
    }
}

```

### Step Five: Create the frontend pod.

This is a simple PHP server that is configured to talk to either the slave or master services depending on whether the request is a read or a write. It exposes a simple AJAX interface, and serves an angular-based UX. Like the redis read slaves it is a replicated service instantiated by a replication controller.

Create a file named `frontend-controller.json`:

```js
{
  "id": "frontendController",
  "desiredState": {
    "replicas": 3,
    "replicasInSet": {"name": "frontend"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "containers": [{
             "image": "brendanburns/php-redis",
             "ports": [{"containerPort": 80, "hostPort": 8080}]
           }]
         }
       },
       "labels": {"name": "frontend"}
      }},
  "labels": {"name": "frontend"}
}
```

With this file, you can turn up your frontend with:

```js
$ cluster/cloudcfg.sh -c examples/guestbook/frontend-controller.json create /replicationControllers

Using master: kubernetes-master (external IP: 1.2.3.4)
{
    "id": "frontendController",
    "desiredState": {
        "replicas": 3,
        "replicasInSet": {
            "name": "frontend"
        },
        "podTemplate": {
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "image": "brendanburns/php-redis",
                            "ports": [
                                {
                                    "hostPort": 8080,
                                    "containerPort": 80
                                }
                            ]
                        }
                    ]
                }
            },
            "labels": {
                "name": "frontend"
            }
        }
    },
    "labels": {
        "name": "frontend"
    }
}

```

Once that's up you can list the pods in the cluster, to verify that the master, slaves and frontends are running:

```js
$ cluster/cloudcfg.sh list /pods

Using master: kubernetes-master (external IP: 1.2.3.4)
{
    "kind": "cluster#podList",
    "items": [
        {
            "id": "redis-master-2",
            "labels": {
                "name": "redis-master"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "name": "master",
                            "image": "dockerfile/redis",
                            "ports": [
                                {
                                    "hostPort": 6379,
                                    "containerPort": 6379
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-3.c.{project-id}.internal"
            }
        },
        {
            "id": "4d65822107fcfd52",
            "labels": {
                "name": "redisslave",
                "replicationController": "redisSlaveController"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "image": "brendanburns/redis-slave",
                            "ports": [
                                {
                                    "hostPort": 6380,
                                    "containerPort": 6379
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-3.c.{project-id}.internal"
            }
        },
        {
            "id": "55104dc76695721d",
            "labels": {
                "name": "frontend",
                "replicationController": "frontendController"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "image": "brendanburns/php-redis",
                            "ports": [
                                {
                                    "hostPort": 8080,
                                    "containerPort": 80
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-3.c.{project-id}.internal"
            }
        },
        {
            "id": "78629a0f5f3f164f",
            "labels": {
                "name": "redisslave",
                "replicationController": "redisSlaveController"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "image": "brendanburns/redis-slave",
                            "ports": [
                                {
                                    "hostPort": 6380,
                                    "containerPort": 6379
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-2.c.{project-id}.internal"
            }
        },
        {
            "id": "380704bb7b4d7c03",
            "labels": {
                "name": "frontend",
                "replicationController": "frontendController"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "image": "brendanburns/php-redis",
                            "ports": [
                                {
                                    "hostPort": 8080,
                                    "containerPort": 80
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-2.c.{project-id}.internal"
            }
        },
        {
            "id": "365a858149c6e2d1",
            "labels": {
                "name": "frontend",
                "replicationController": "frontendController"
            },
            "desiredState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": [
                        {
                            "image": "brendanburns/php-redis",
                            "ports": [
                                {
                                    "hostPort": 8080,
                                    "containerPort": 80
                                }
                            ]
                        }
                    ]
                }
            },
            "currentState": {
                "manifest": {
                    "version": "",
                    "volumes": null,
                    "containers": null
                },
                "host": "kubernetes-minion-4.c.{project-id}.internal"
            }
        }
    ]
}

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

To play with the service itself, find the name of a frontend, grab the external IP of that host from the [Google Cloud Console][cloud-console], and visit `http://<host-ip>:8080`. You may need to open the firewall for port 8080 using the [console][cloud-console] or the `gcutil` tool.

[cloud-console]: https://console.developer.google.com
