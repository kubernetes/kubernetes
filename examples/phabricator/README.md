<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Phabricator example

This example shows how to build a simple multi-tier web application using Kubernetes and Docker.

The example combines a web frontend and an external service that provides MySQL database. We use CloudSQL on Google Cloud Platform in this example, but in principle any approach to running MySQL should work.

### Step Zero: Prerequisites

This example assumes that you have a basic understanding of kubernetes [services](../../docs/user-guide/services.md) and that you have forked the repository and [turned up a Kubernetes cluster](../../docs/getting-started-guides/):

```sh
$ cd kubernetes
$ hack/dev-build-and-up.sh
```

### Step One: Set up Cloud SQL instance

Follow the [official instructions](https://cloud.google.com/sql/docs/getting-started) to set up Cloud SQL instance.

In the remaining part of this example we will assume that your instance is named "phabricator-db", has IP 173.194.242.66 and the password is "1234".

### Step Two: Turn up the phabricator

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

### Step Three: Authenticate phabricator in Cloud SQL

If you read logs of the phabricator container you will notice the following error message:

```bash
$ kubectl logs phabricator-controller-02qp4
[...]
Raw MySQL Error: Attempt to connect to root@173.194.252.142 failed with error
#2013: Lost connection to MySQL server at 'reading initial communication packet', system error: 0.

```

This is because the host on which this container is running is not authorized in Cloud SQL. To fix this run:

```bash
gcloud sql instances patch phabricator-db --authorized-networks 130.211.141.151
```

To automate this process and make sure that a proper host is authorized even if pod is rescheduled to a new machine we need a separate pod that periodically lists pods and authorizes hosts. Use the file [`examples/phabricator/authenticator-controller.json`](authenticator-controller.json):

<!-- BEGIN MUNGE: EXAMPLE authenticator-controller.json -->

```json
{
  "kind": "ReplicationController",
  "apiVersion": "v1",
  "metadata": {
    "name": "authenticator-controller",
    "labels": {
      "name": "authenticator"
    }
  },
  "spec": {
    "replicas": 1,
    "selector": {
      "name": "authenticator"
    },
    "template": {
      "metadata": {
        "labels": {
          "name": "authenticator"
        }
      },
      "spec": {
        "containers": [
          {
            "name": "authenticator",
            "image": "gcr.io/google_containers/cloudsql-authenticator:v1"
          }
        ]
      }
    }
  }
}
```

[Download example](authenticator-controller.json?raw=true)
<!-- END MUNGE: EXAMPLE authenticator-controller.json -->

To create the pod run:

```sh
$ kubectl create -f examples/phabricator/authenticator-controller.json
```


### Step Four: Turn up the phabricator service

A Kubernetes 'service' is a named load balancer that proxies traffic to one or more containers. The services in a Kubernetes cluster are discoverable inside other containers via *environment variables*. Services find the containers to load balance based on pod labels.  These environment variables are typically referenced in application code, shell scripts, or other places where one node needs to talk to another in a distributed system.  You should catch up on [kubernetes services](../../docs/user-guide/services.md) before proceeding.

The pod that you created in Step One has the label `name=phabricator`. The selector field of the service determines which pods will receive the traffic sent to the service. Since we are setting up a service for an external application we also need to request external static IP address (otherwise it will be assigned dynamically):

```sh
$ gcloud compute addresses create phabricator --region us-central1
Created [https://www.googleapis.com/compute/v1/projects/myproject/regions/us-central1/addresses/phabricator].
NAME         REGION      ADDRESS        STATUS
phabricator  us-central1 107.178.210.6  RESERVED
```

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

To play with the service itself, find the `EXTERNAL_IP` of the load balancer:

```console
$ kubectl get services phabricator
NAME                  CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
phabricator           10.0.0.2         1.2.3.4           8080/TCP      ...                    ...
```

and then visit port 80 of that IP address.

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
