<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/examples/celery-rabbitmq/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

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

You should already have turned up a Kubernetes cluster. To get the most of this example, ensure that Kubernetes will create more than one node (e.g. by setting your `NUM_NODES` environment variable to 2 or more).


## Step 1: Start the RabbitMQ service

The Celery task queue will need to communicate with the RabbitMQ broker. RabbitMQ will eventually appear on a separate pod, but since pods are ephemeral we need a service that can transparently route requests to RabbitMQ.

<!-- BEGIN MUNGE: EXAMPLE rabbitmq-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    component: rabbitmq
  name: rabbitmq-service
spec:
  ports:
  - port: 5672
  selector:
    app: taskQueue
    component: rabbitmq
```

[Download example](rabbitmq-service.yaml?raw=true)
<!-- END MUNGE: EXAMPLE rabbitmq-service.yaml -->

To start the service, run:

```sh
$ kubectl create -f examples/celery-rabbitmq/rabbitmq-service.yaml
```

This service allows other pods to connect to the rabbitmq. To them, it will be seen as available on port 5672, although the service is routing the traffic to the container (also via port 5672).


## Step 2: Fire up RabbitMQ

A RabbitMQ broker can be turned up using the file [`examples/celery-rabbitmq/rabbitmq-controller.yaml`](rabbitmq-controller.yaml):

<!-- BEGIN MUNGE: EXAMPLE rabbitmq-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  labels:
    component: rabbitmq
  name: rabbitmq-controller
spec:
  replicas: 1
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
        resources:
          limits:
            cpu: 100m
        livenessProbe:
          httpGet:
            # Path to probe; should be cheap, but representative of typical behavior
            path: /
            port: 5672
          initialDelaySeconds: 30
          timeoutSeconds: 1
```

[Download example](rabbitmq-controller.yaml?raw=true)
<!-- END MUNGE: EXAMPLE rabbitmq-controller.yaml -->

Running `$ kubectl create -f examples/celery-rabbitmq/rabbitmq-controller.yaml` brings up a replication controller that ensures one pod exists which is running a RabbitMQ instance.

Note that bringing up the pod includes pulling down a docker image, which may take a few moments. This applies to all other pods in this example.


## Step 3: Fire up Celery

Bringing up the celery worker is done by running `$ kubectl create -f examples/celery-rabbitmq/celery-controller.yaml`, which contains this:

<!-- BEGIN MUNGE: EXAMPLE celery-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  labels:
    component: celery
  name: celery-controller
spec:
  replicas: 1
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
        resources:
          limits:
            cpu: 100m
```

[Download example](celery-controller.yaml?raw=true)
<!-- END MUNGE: EXAMPLE celery-controller.yaml -->

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

The celery\_conf.py contains the definition of a simple Celery task that adds two numbers. This last line starts the Celery worker.

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

First, start the flower service with `$ kubectl create -f examples/celery-rabbitmq/flower-service.yaml`. The service is defined as below:

<!-- BEGIN MUNGE: EXAMPLE flower-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    component: flower
  name: flower-service
spec:
  ports:
  - port: 5555
  selector:
    app: taskQueue
    component: flower
  type: LoadBalancer
```

[Download example](flower-service.yaml?raw=true)
<!-- END MUNGE: EXAMPLE flower-service.yaml -->

It is marked as external (LoadBalanced). However on many platforms you will have to add an explicit firewall rule to open port 5555.
On GCE this can be done with:

```
 $ gcloud compute firewall-rules create --allow=tcp:5555 --target-tags=kubernetes-minion kubernetes-minion-5555
```

Please remember to delete the rule after you are done with the example (on GCE: `$ gcloud compute firewall-rules delete kubernetes-minion-5555`)

To bring up the pods, run this command `$ kubectl create -f examples/celery-rabbitmq/flower-controller.yaml`. This controller is defined as so:

<!-- BEGIN MUNGE: EXAMPLE flower-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  labels:
    component: flower
  name: flower-controller
spec:
  replicas: 1
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
        resources:
          limits:
            cpu: 100m
        livenessProbe:
          httpGet:
            # Path to probe; should be cheap, but representative of typical behavior
            path: /
            port: 5555
          initialDelaySeconds: 30
          timeoutSeconds: 1
```

[Download example](flower-controller.yaml?raw=true)
<!-- END MUNGE: EXAMPLE flower-controller.yaml -->

This will bring up a new pod with Flower installed and port 5555 (Flower's default port) exposed through the service endpoint. This image uses the following command to start Flower:

```sh
flower --broker=amqp://guest:guest@${RABBITMQ_SERVICE_SERVICE_HOST:localhost}:5672//
```

Again, it uses the Kubernetes-provided environment variable to obtain the address of the RabbitMQ service.

Once all pods are up and running, running `kubectl get pods` will display something like this:

```
NAME                                           READY     REASON       RESTARTS   AGE
celery-controller-wqkz1                        1/1       Running      0          8m
flower-controller-7bglc                        1/1       Running      0          7m
rabbitmq-controller-5eb2l                      1/1       Running      0          13m
```

`kubectl get service flower-service` will help you to get the external IP addresses of the flower service.

```
NAME             LABELS             SELECTOR                         IP(S)            PORT(S)
flower-service   component=flower   app=taskQueue,component=flower   10.0.44.166      5555/TCP
                                                                162.222.181.180
```

Point your internet browser to the appropriate flower-service address, port 5555 (in our case http://162.222.181.180:5555).
If you click on the tab called "Tasks", you should see an ever-growing list of tasks called "celery_conf.add" which the run\_tasks.py script is dispatching.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/celery-rabbitmq/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
