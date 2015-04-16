## Getting started on Microsoft Azure

### Azure Prerequisites

1. You need an Azure account. Visit http://azure.microsoft.com/ to get started.
2. Install and configure the Azure cross-platform command-line interface. http://azure.microsoft.com/en-us/documentation/articles/xplat-cli/
3. Make sure you have a default account set in the Azure cli, using `azure account set`

### Prerequisites for your workstation

1. Be running a Linux or Mac OS X.
2. Get or build a [binary release](binary_release.md)
3. If you want to build your own release, you need to have [Docker
installed](https://docs.docker.com/installation/).  On Mac OS X you can use
[boot2docker](http://boot2docker.io/).

### Setup
The cluster setup scripts can setup Kubernetes for multiple targets. First modify `cluster/kube-env.sh` to specify azure:

    KUBERNETES_PROVIDER="azure"

Next, specify an existing virtual network and subnet in `cluster/azure/config-default.sh`:

    AZ_VNET=<vnet name>
    AZ_SUBNET=<subnet name>

You can create a virtual network:

    azure network vnet create <vnet name> --subnet=<subnet name> --location "West US" -v

Now you're ready.

You can then use the `cluster/kube-*.sh` scripts to manage your azure cluster, start with:

    cluster/kube-up.sh

The script above will start (by default) a single master VM along with 4 worker VMs.  You
can tweak some of these parameters by editing `cluster/azure/config-default.sh`.

### Running a container (simple version)

Once you have your instances up and running, the `hack/build-go.sh` script sets up
your Go workspace and builds the Go components.

The `kubectl.sh` line below spins up two containers running
[Nginx](http://nginx.org/en/) running on port 80:

```bash
cluster/kubectl.sh run-container my-nginx --image=nginx --replicas=2 --port=80
```

To stop the containers:

```bash
cluster/kubectl.sh stop rc my-nginx
```

To delete the containers:

```bash
cluster/kubectl.sh delete rc my-nginx
```

### Running a container (more complete version)


You can create a pod like this:


```
cd kubernetes
cluster/kubectl.sh create -f docs/getting-started-guides/pod.json
```

Where pod.json contains something like:

```
{
  "id": "php",
  "kind": "Pod",
  "apiVersion": "v1beta1",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "php",
      "containers": [{
        "name": "nginx",
        "image": "nginx",
        "ports": [{
          "containerPort": 80,
          "hostPort": 8080
        }],
        "livenessProbe": {
          "enabled": true,
          "type": "http",
          "initialDelaySeconds": 30,
          "httpGet": {
            "path": "/index.html",
            "port": 8080
          }
        }
      }]
    }
  },
  "labels": {
    "name": "foo"
  }
}
```

You can see your cluster's pods:

```
cluster/kubectl.sh get pods
```

and delete the pod you just created:

```
cluster/kubectl.sh delete pods php
```

Look in `api/examples/` for more examples

### Tearing down the cluster
```
cluster/kube-down.sh
```
