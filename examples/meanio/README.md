## MEAN.io example

This example shows how to build a simple, multi-tier web application using Kubernetes and Docker.
It is based on the [guestbook](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/guestbook) example.

The example consists of:
- A nodejs web app using MEAN framework.
- A MongoDB server.

### Step Zero: Prerequisites

This example requires a kubernetes cluster. use the prerequisites of the [guestbook](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/guestbook#step-zero-prerequisites) example.

### Step One: Fire up the MongoDB Server

Use (or just create) the file `examples/meanio/meanio-mongodb-controller.json` which describes a single pod running a MongoDB Server:
Note that, although the MongoDB server runs just with a single replica, we use replication controller to enforce that exactly one pod keeps running (e.g. in a event of node going down, the replication controller will ensure that the MognoDB server gets restarted on a healthy node).   This could result in data loss.
We are using mongodb 2.6.8 for MEAN.io compatibilily.


```js
{
  "id": "meanio-mongodb-controller",
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "desiredState": {
    "replicas": 1,
    "replicaSelector": {"name": "meanio-mongodb"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "meanio-mongodb",
           "containers": [{
             "name": "meanio-mongodb",
             "image": "mongo:2.6.8",
             "ports": [{"containerPort": 27017}]
           }]
         }
      },
      "labels": {
        "name": "meanio-mongodb",
        "app": "mongodb"
      }
    }
  },
  "labels": {"name": "meanio-mongodb"}
}
```

Now, create the MongoDB pod in your Kubernetes cluster by running:

```shell
$ kubectl create -f examples/meanio/meanio-mongodb-controller.json

$ cluster/kubectl.sh get rc
CONTROLLER                             CONTAINER(S)            IMAGE(S)                                 SELECTOR                     REPLICAS
meanio-mongodb-controller              meanio-mongodb          mongo:2.6.8                         name=meanio-mongodb          1
```

Once that's up you can list the pods in the cluster, to verify that the MongoDB server is running:


```shell
$ kubectl get pods
```

You'll see all kubernetes components, most importantly the MongoDB server pod. It will also display the machine that the pod is running on once it gets placed:

```shell
POD                                          IP                  CONTAINER(S)            IMAGE(S)                                 HOST                                                              LABELS                                                     STATUS
meanio-mongodb-controller-nyxxv              10.244.0.11         meanio-mongodb          mongo:2.6.8                         kubernetes-minion-yxea.c.mean-io.internal/104.155.36.70   app=mongodb,name=meanio-mongodb
```

If you ssh to that machine, you can run `docker ps` to see the actual pod:

```shell
me@workstation$ gcloud compute ssh kubernetes-minion-yxea

me@kubernetes-minion-yxea:~$ sudo docker ps
CONTAINER ID        IMAGE                                  COMMAND                CREATED              STATUS              PORTS                    NAMES
efabcc7f7da9        mongo:2                                "/entrypoint.sh mong   23 hours ago        Up 23 hours                                                          k8s_meanio-mongodb.e5bffb35_meanio-mongodb-controller-nyxxv.default.api_499dd2a6-ce04-11e4-94ac-42010af070a6_d9391efe
```

(Note that initial `docker pull` may take a few minutes, depending on network conditions.  You can monitor the status of this by running `journalctl -f -u docker` to check when the image is being downloaded.  Of course, you can also run `journalctl -f -u kubelet` to see what state the kubelet is in as well during this time.

### Step Two: Fire up the MongoDB service
A Kubernetes 'service' is a named load balancer that proxies traffic to *one or more* containers. This is done using the *labels* metadata which we defined in the meanio-mongodb pod above.

The services in a Kubernetes cluster are discoverable inside other containers via environment variables.

Services find the containers to load balance based on pod labels.

The pod that you created in Step One has the label `name=meanio-mongodb`. The selector field of the service determines *which pods will receive the traffic* sent to the service, and the port and containerPort information defines what port the service proxy will run at. 

Use the file `examples/meanio/meanio-mongodb-service.json`:

```js
{
  "id": "meanio-mongodb",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 27017,
  "containerPort": 27017,
  "labels": {
    "name": "meanio-mongodb"
  },
  "selector": {
    "name": "meanio-mongodb"
  }
}
```

to create the service by running:

```shell
$ kubectl create -f examples/meanio/meanio-mongodb-service.json
meanio-mongodb

$ kubectl get services
NAME                    LABELS                                    SELECTOR                     IP                  PORT
meanio-mongodb          name=meanio-mongodb                                             name=meanio-mongodb          10.0.222.168        27017
```

This will cause all pods to see the MongoDB server apparently running on <ip>:27017.

- Traffic will be forwarded from the service "port" (on the service node) to the  *containerPort* on the pod which (a node the service listens to). 

Thus, once created, the service proxy on each minion is configured to set up a proxy on the specified port (in this case port 27017).


Make sure that the mongodb pod is running and listening by running:

```shell

$ kubectl log meanio-mongodb-controller-nyxxv

```

### Step Three: Create the meanio pod

This is a nodejs server that is configured to talk to the MognoDB server. 
Note that you will need the MongoDB server up and running before running the meanio pod.

The pod is described in the file `examples/meanio/meanio-controller.json`:

```js
{
  "id": "meanio-controller",
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "desiredState": {
    "replicas": 1,
    "replicaSelector": {"name": "meanio"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "meanio",
           "containers": [{
             "name": "meanio",
             "image":"shaiweinstein/k8s:v1",
             "ports": [{"name": "meanio-server", "containerPort": 3000}],
             "env": [{"name": "DB_PORT_27017_TCP_ADDR", "value": "meanio-mongodb" }]
           }]
         }
       },
       "labels": {
         "name": "meanio",
         "uses": "meanio-mongodb",
         "app": "meanio"
       }
      }},
  "labels": {"name": "meanio"}
}
```

Using this file, you can turn up your meanio framework  with:

```shell
$ kubectl create -f examples/meanio/meanio-controller.json
meanio-controller

$ kubectl get rc
CONTROLLER                             CONTAINER(S)            IMAGE(S)                                   SELECTOR                     REPLICAS
meanio-controller                      meanio                  shaiweinstein/k8s:v1                name=meanio                  1
meanio-mongodb-controller              meanio-mongodb          mongo:2.6.8                         name=meanio-mongodb          1

```

Once that's up (it may take a few minutes to create the pod) you can list the pods, to verify that the mongodb and meanio are running:

```shell
$ kubectl get pods
POD                                          IP                  CONTAINER(S)            IMAGE(S)                                   HOST                                                              LABELS                                                     STATUS
meanio-controller-zwk1b                      10.244.0.13         meanio                  shaiweinstein/k8s:v1                kubernetes-minion-yxea.c.mean-io.internal/104.155.36.70   app=meanio,name=meanio,uses=meanio-mongodb                                  Running             23 hours
meanio-mongodb-controller-nyxxv              10.244.0.11         meanio-mongodb          mongo:2.6.8                         kubernetes-minion-yxea.c.mean-io.internal/104.155.36.70   app=mongodb,name=meanio-mongodb                                             Running             23 hours

```

You will see a mongodb server pod, and meanio pod.

### Step Four: Create the meanio service

Just like the others, you want a service to group your meanio pods (in case of running more than one relica)..
The service is described in the file `examples/meanio/meanio-service.json`:

```js
{
  "id": "meanio",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 80,
  "containerPort": "meanio-server",
  "selector": {
    "name": "meanio"
  },
  "labels": {
    "name": "meanio"
  },
  "createExternalLoadBalancer": true
}
```

If running a single node local setup, or single VM, you don't need `createExternalLoadBalancer`.

```shell
$ kubectl create -f examples/meanio/meanio-service.json
meanio


$ kubectl get services
NAME                    LABELS                                    SELECTOR                     IP                  PORT
meanio                  name=meanio                                                     name=meanio                  10.0.29.25          80
meanio-mongodb          name=meanio-mongodb                                             name=meanio-mongodb          10.0.222.168        27017
```

### A few Google Container Engine specifics for playing around with the services.

In GCE, `cluster/kubectl.sh` automatically creates forwarding rule for services with `createExternalLoadBalancer`.

```shell
$ gcloud compute forwarding-rules list
NAME                  REGION      IP_ADDRESS     IP_PROTOCOL TARGET
kubernetes-meanio europe-west1 130.211.xxx.xxx TCP         europe-west1/targetPools/kubernetes-meanio
```

You can grab the external IP of the load balancer associated with that rule and visit `http://130.211.xxx.xxx`.

In GCE, you also may need to open the firewall for port 80 using the [console][cloud-console] or the `gcloud` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion`:

```shell
$ gcloud compute firewall-rules create --allow=tcp:80 --target-tags=kubernetes-minion kubernetes-minion-80
```

For GCE details about limiting traffic to specific sources, see the [GCE firewall documentation][gce-firewall-docs].

[cloud-console]: https://console.developer.google.com
[gce-firewall-docs]: https://cloud.google.com/compute/docs/networking#firewalls
