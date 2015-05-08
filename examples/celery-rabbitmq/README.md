# Example: Distributed task queues with Celery, RabbitMQ and Flower

## Introduction

Celery is an asynchronous task queue based on distributed message passing. It is used to create execution units (i.e. tasks) which are then executed on one or more worker nodes, either synchronously or asynchronously. 

Celery is implemented in Python.

Since Celery is based on message passing, it requires some middleware (to handle translation of the message between sender and receiver) called a _message broker_. RabbitMQ is a message broker often used in conjunction with Celery.

This example will show you how to use Kubernetes to set up a very basic distributed task queue using Celery as the task queue and RabbitMQ as the message broker. It will also show you how to set up a Flower-based front end to monitor the tasks.

## Goal

At the end of the example, we will have:

* Three pods:
    * A Celery task queue
    * A RabbitMQ message broker
    * A Flower frontend
* A service that provides access to the message broker
* A basic celery task that can be passed to the worker node


## Prerequisites

You should already have turned up a Kubernetes cluster. To get the most of this example, ensure that Kubernetes will create more than one minion (e.g. by setting your `NUM_MINIONS` environment variable to 2 or more).


## Step 1: Start the RabbitMQ service

The Celery task queue will need to communicate with the RabbitMQ broker. RabbitMQ will eventually appear on a separate pod, but since pods are ephemeral we need a service that can transparently route requests to RabbitMQ.

Use the file `examples/celery-rabbitmq/rabbitmq-service.yaml`:

```yaml
apiVersion: v1beta3
kind: Service
metadata:
  labels:
    name: rabbitmq
  name: rabbitmq-service
spec:
  ports:
  - port: 5672
    protocol: TCP
    targetPort: 5672
  selector:
    app: taskQueue
    component: rabbitmq
```

To start the service, run:

```shell
$ kubectl create -f examples/celery-rabbitmq/rabbitmq-service.yaml
```

**NOTE**: If you're running Kubernetes from source, you can use `cluster/kubectl.sh` instead of `kubectl`.

This service allows other pods to connect to the rabbitmq. To them, it will be seen as available on port 5672, although the service is routing the traffic to the container (also via port 5672).


## Step 2: Fire up RabbitMQ

A RabbitMQ broker can be turned up using the file `examples/celery-rabbitmq/rabbitmq-controller.yaml`:

```yaml
apiVersion: v1beta3
kind: ReplicationController
metadata:
  labels:
    name: rabbitmq
  name: rabbitmq-controller
spec:
  replicas: 1
  selector:
    component: rabbitmq
  template:
    metadata:
      labels:
        app: taskQueue
        component: rabbitmq
    spec:
      containers:
      - image: rabbitmq
        name: rabbitmq
        ports:
        - containerPort: 5672
          protocol: TCP
        resources:
          limits:
            cpu: 100m
```

Running `$ kubectl create -f examples/celery-rabbitmq/rabbitmq-controller.yaml` brings up a replication controller that ensures one pod exists which is running a RabbitMQ instance.

Note that bringing up the pod includes pulling down a docker image, which may take a few moments. This applies to all other pods in this example.


## Step 3: Fire up Celery

Bringing up the celery worker is done by running `$ kubectl create -f examples/celery-rabbitmq/celery-controller.yaml`, which contains this:

```yaml
apiVersion: v1beta3
kind: ReplicationController
metadata:
  labels:
    name: celery
  name: celery-controller
spec:
  replicas: 1
  selector:
    component: celery
  template:
    metadata:
      labels:
        app: taskQueue
        component: celery
    spec:
      containers:
      - image: endocode/celery-app-add
        name: celery
        ports:
        - containerPort: 5672
          protocol: TCP
        resources:
          limits:
            cpu: 100m
```

There are several things to point out here...

Like the RabbitMQ controller, this controller ensures that there is always a pod is running a Celery worker instance. The celery-app-add Docker image is an extension of the standard Celery image. This is the Dockerfile:

```
FROM library/celery

ADD celery_conf.py /data/celery_conf.py
ADD run_tasks.py /data/run_tasks.py
ADD run.sh /usr/local/bin/run.sh

ENV C_FORCE_ROOT 1

CMD ["/bin/bash", "/usr/local/bin/run.sh"]
```

The celery\_conf.py contains the defintion of a simple Celery task that adds two numbers. This last line starts the Celery worker.

**NOTE:** `ENV C_FORCE_ROOT 1` forces Celery to be run as the root user, which is *not* recommended in production!

The celery\_conf.py file contains the following:

```python
import os

from celery import Celery

# Get Kubernetes-provided address of the broker service
broker_service_host = os.environ.get('RABBITMQ_SERVICE_SERVICE_HOST')

app = Celery('tasks', broker='amqp://guest@%s//' % broker_service_host, backend='amqp')

@app.task
def add(x, y):
    return x + y
```

Assuming you're already familiar with how Celery works, everything here should be familiar, except perhaps the part `os.environ.get('RABBITMQ_SERVICE_SERVICE_HOST')`. This environment variable contains the IP address of the RabbitMQ service we created in step 1. Kubernetes automatically provides this environment variable to all containers which have the same app label as that defined in the RabbitMQ service (in this case "taskQueue"). In the Python code above, this has the effect of automatically filling in the broker address when the pod is started.

The second python script (run\_tasks.py) periodically executes the `add()` task every 5 seconds with a couple of random numbers.

The question now is, how do you see what's going on?


## Step 4: Put a frontend in place

Flower is a web-based tool for monitoring and administrating Celery clusters. By connecting to the node that contains Celery, you can see the behaviour of all the workers and their tasks in real-time.

To bring up the frontend, run this command `$ kubectl create -f examples/celery-rabbitmq/flower-controller.yaml`. This controller is defined as so:

```yaml
apiVersion: v1beta3
kind: ReplicationController
metadata:
  labels:
    name: flower
  name: flower-controller
spec:
  replicas: 1
  selector:
    component: flower
  template:
    metadata:
      labels:
        app: taskQueue
        component: flower
    spec:
      containers:
      - image: endocode/flower
        name: flower
        ports:
        - containerPort: 5555
          hostPort: 5555
          protocol: TCP
        resources:
          limits:
            cpu: 100m
```

This will bring up a new pod with Flower installed and port 5555 (Flower's default port) exposed. This image uses the following command to start Flower:

```sh
flower --broker=amqp://guest:guest@${RABBITMQ_SERVICE_SERVICE_HOST:localhost}:5672//
```

Again, it uses the Kubernetes-provided environment variable to obtain the address of the RabbitMQ service.

Once all pods are up and running, running `kubectl get pods` will display something like this:

```
POD                         IP                  CONTAINER(S)        IMAGE(S)                           HOST                    LABELS                                                STATUS
celery-controller-h3x9k     10.246.1.11         celery              endocode/celery-app-add            10.245.1.3/10.245.1.3   app=taskQueue,name=celery                             Running
flower-controller-cegta     10.246.2.17         flower              endocode/flower                    10.245.1.4/10.245.1.4   app=taskQueue,name=flower                             Running
kube-dns-fplln              10.246.1.3          etcd                quay.io/coreos/etcd:latest         10.245.1.3/10.245.1.3   k8s-app=kube-dns,kubernetes.io/cluster-service=true   Running
                                                kube2sky            kubernetes/kube2sky:1.0                                                                                          
                                                skydns              kubernetes/skydns:2014-12-23-001                                                                                 
rabbitmq-controller-pjzb3   10.246.2.16         rabbitmq            library/rabbitmq                   10.245.1.4/10.245.1.4   app=taskQueue,name=rabbitmq                           Running

```

Now you know on which host Flower is running (in this case, 10.245.1.4), you can open your browser and enter the address (e.g. `http://10.245.1.4:5555`. If you click on the tab called "Tasks", you should see an ever-growing list of tasks called "celery_conf.add" which the run\_tasks.py script is dispatching.

