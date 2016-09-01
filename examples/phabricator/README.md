<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Phabricator example

This example shows how to build a simple multi-tier web application using Kubernetes and Docker.

The example combines a web frontend and an external service that provides MySQL database. We use CloudSQL on Google Cloud Platform in this example, but in principle any approach to running MySQL should work.

### Step Zero: Prerequisites

This example assumes that you have a basic understanding of kubernetes [services](../../docs/user-guide/services.md) and that you have forked the repository and [turned up a Kubernetes cluster](../../docs/getting-started-guides/):

```sh
$ cd kubernetes
$ cluster/kube-up.sh
```

### Step One: Set up Cloud SQL instance

Follow the [official instructions](https://cloud.google.com/sql/docs/getting-started) to set up Cloud SQL instance.

In the remaining part of this example we will assume that your instance is named "phabricator-db", has IP 1.2.3.4, is listening on port 3306 and the password is "1234".

### Step Two: Authenticate phabricator in Cloud SQL

In order to allow phabricator to connect to your Cloud SQL instance you need to run the following command to authorize all your nodes within a cluster:

```bash
NODE_NAMES=`kubectl get nodes | cut -d" " -f1 | tail -n+2`
NODE_IPS=`gcloud compute instances list $NODE_NAMES | tr -s " " | cut -d" " -f 5 | tail -n+2`
gcloud sql instances patch phabricator-db --authorized-networks $NODE_IPS
```

Otherwise you will see the following logs:

```bash
$ kubectl logs phabricator-controller-02qp4
[...]
Raw MySQL Error: Attempt to connect to root@1.2.3.4 failed with error
#2013: Lost connection to MySQL server at 'reading initial communication packet', system error: 0.

```

### Step Three: Turn up the phabricator

To start Phabricator server use the file [`examples/phabricator/phabricator-controller.json`](phabricator-controller.json) which describes a [replication controller](../../docs/user-guide/replication-controller.md) with a single [pod](../../docs/user-guide/pods.md) running an Apache server with Phabricator PHP source:

<!-- BEGIN MUNGE: EXAMPLE phabricator-controller.json -->

```json
{
  "kind": "ReplicationController",
  "apiVersion": "v1",
  "metadata": {
    "name": "phabricator-controller",
    "labels": {
      "name": "phabricator"
    }
  },
  "spec": {
    "replicas": 1,
    "selector": {
      "name": "phabricator"
    },
    "template": {
      "metadata": {
        "labels": {
          "name": "phabricator"
        }
      },
      "spec": {
        "containers": [
          {
            "name": "phabricator",
            "image": "fgrzadkowski/example-php-phabricator",
            "ports": [
              {
                "name": "http-server",
                "containerPort": 80
              }
            ],
            "env": [
              {
                "name": "MYSQL_SERVICE_IP",
                "value": "1.2.3.4"
              },
              {
                "name": "MYSQL_SERVICE_PORT",
                "value": "3306"
              },
              {
                "name": "MYSQL_PASSWORD",
                "value": "1234"
              }
            ]
          }
        ]
      }
    }
  }
}
```

[Download example](phabricator-controller.json?raw=true)
<!-- END MUNGE: EXAMPLE phabricator-controller.json -->

Create the phabricator pod in your Kubernetes cluster by running:

```sh
$ kubectl create -f examples/phabricator/phabricator-controller.json
```

**Note:** Remember to substitute environment variable values in json file before create replication controller.

Once that's up you can list the pods in the cluster, to verify that it is running:

```sh
kubectl get pods
```

You'll see a single phabricator pod. It will also display the machine that the pod is running on once it gets placed (may take up to thirty seconds):

```
NAME                           READY     STATUS    RESTARTS   AGE
phabricator-controller-9vy68   1/1       Running   0          1m
```

If you ssh to that machine, you can run `docker ps` to see the actual pod:

```sh
me@workstation$ gcloud compute ssh --zone us-central1-b kubernetes-minion-2

$ sudo docker ps
CONTAINER ID        IMAGE                             COMMAND     CREATED       STATUS      PORTS   NAMES
54983bc33494        fgrzadkowski/phabricator:latest   "/run.sh"   2 hours ago   Up 2 hours          k8s_phabricator.d6b45054_phabricator-controller-02qp4.default.api_eafb1e53-b6a9-11e4-b1ae-42010af05ea6_01c2c4ca
```

(Note that initial `docker pull` may take a few minutes, depending on network conditions.  During this time, the `get pods` command will return `Pending` because the container has not yet started )

### Step Four: Turn up the phabricator service

A Kubernetes 'service' is a named load balancer that proxies traffic to one or more containers. The services in a Kubernetes cluster are discoverable inside other containers via *environment variables*. Services find the containers to load balance based on pod labels.  These environment variables are typically referenced in application code, shell scripts, or other places where one node needs to talk to another in a distributed system.  You should catch up on [kubernetes services](../../docs/user-guide/services.md) before proceeding.

The pod that you created in Step Three has the label `name=phabricator`. The selector field of the service determines which pods will receive the traffic sent to the service.

Use the file [`examples/phabricator/phabricator-service.json`](phabricator-service.json):

<!-- BEGIN MUNGE: EXAMPLE phabricator-service.json -->

```json
{
  "kind": "Service",
  "apiVersion": "v1",
  "metadata": {
    "name": "phabricator"
  },
  "spec": {
    "ports": [
      {
        "port": 80,
        "targetPort": "http-server"
      }
    ],
    "selector": {
      "name": "phabricator"
    },
    "type": "LoadBalancer"
  }
}
```

[Download example](phabricator-service.json?raw=true)
<!-- END MUNGE: EXAMPLE phabricator-service.json -->

To create the service run:

```sh
$ kubectl create -f examples/phabricator/phabricator-service.json
phabricator
```

To play with the service itself, find the external IP of the load balancer:

```console
$ kubectl get services
NAME          LABELS                                    SELECTOR           IP(S)         PORT(S)
kubernetes    component=apiserver,provider=kubernetes   <none>             10.0.0.1      443/TCP
phabricator   <none>                                    name=phabricator   10.0.31.173   80/TCP
$ kubectl get services phabricator -o json | grep ingress -A 4
            "ingress": [
                {
                    "ip": "104.197.13.125"
                }
            ]
```

and then visit port 80 of that IP address.

**Note**: Provisioning of the external IP address may take few minutes.

**Note**: You may need to open the firewall for port 80 using the [console][cloud-console] or the `gcloud` tool. The following command will allow traffic from any source to instances tagged `kubernetes-minion`:

```sh
$ gcloud compute firewall-rules create phabricator-node-80 --allow=tcp:80 --target-tags kubernetes-minion
```

### Step Six: Cleanup

To turn down a Kubernetes cluster:

```sh
$ cluster/kube-down.sh
```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/phabricator/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
