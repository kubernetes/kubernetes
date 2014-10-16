# WARNING
These instructions are broken at git HEAD.  Please either:
* Sync back to `v0.3` with `git checkout v0.3`
* Download a [snapshot of `v0.3`](https://github.com/GoogleCloudPlatform/kubernetes/archive/v0.3.tar.gz)

## Getting started on Microsoft Azure

### Prerequisites

1. You need an Azure account. Visit http://azure.microsoft.com/ to get started.
2. Install and configure the Azure cross-platform command-line interface. http://azure.microsoft.com/en-us/documentation/articles/xplat-cli/
3. Make sure you have a default account set in the Azure cli, using `azure account set`
4. You must have Go (version 1.2 or later) installed: [www.golang.org](http://www.golang.org).
5. Get the Kubernetes source:

        git clone https://github.com/GoogleCloudPlatform/kubernetes.git

### Setup
The cluster setup scripts can setup Kubernetes for multiple targets. First modify `cluster/kube-env.sh` to specify azure:

    KUBERNETES_PROVIDER="azure"

Next build Kubernetes, package the release, and upload to Azure Storage:

    cd kubernetes
    release/azure/release.sh

Next, specify an existing virtual network in `cluster/azure/config-defualt.sh`:

    AZ_VNET=<vnet name>

Now you're ready.

You can then use the `cluster/kube-*.sh` scripts to manage your azure cluster, start with:

    cluster/kube-up.sh

### Running a container (simple version)

Once you have your instances up and running, the `hack/build-go.sh` script sets up
your Go workspace and builds the Go components.

The `cluster/kubecfg.sh` script spins up two containers, running [Nginx](http://nginx.org/en/) and with port 80 mapped to 8080:

```
cd kubernetes
hack/build-go.sh
cluster/kubecfg.sh -p 8080:80 run dockerfile/nginx 2 myNginx
```

To stop the containers:
```
cluster/kubecfg.sh stop myNginx
```

To delete the containers:
```
cluster/kubecfg.sh rm myNginx
```

### Running a container (more complete version)


You can create a pod like this:


```
cd kubernetes
cluster/kubecfg.sh -c api/examples/pod.json create /pods
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
        "image": "dockerfile/nginx",
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
            "port": "8080"
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
cluster/kubecfg.sh list pods
```

and delete the pod you just created:

```
cluster/kubecfg.sh delete pods/php
```

Look in `api/examples/` for more examples

### Tearing down the cluster
```
cd kubernetes
cluster/kube-down.sh
```
