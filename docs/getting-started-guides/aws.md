## Getting started on AWS

### Prerequisites

1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)
3. You must have Go (version 1.2 or later) installed: [www.golang.org](http://www.golang.org).
4. Get the Kubernetes source:

        git clone https://github.com/GoogleCloudPlatform/kubernetes.git

### Setup

The setup script builds Kubernetes, then creates AWS VPC, instances, firewall rules, and routes:

```
export KUBERNETES_PROVIDER=aws
cd kubernetes
hack/aws/dev-build-and-up.sh
```

The script above relies on AWS S3 to deploy the software to instances running in EC2.

### Running a container (simple version)

Once you have your instances up and running, the `build-go.sh` script sets up
your Go workspace and builds the Go components.

The `kubecfg.sh` script spins up two containers, running [Nginx](http://nginx.org/en/) and with port 80 mapped to 8080:

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


Assuming you've run `hack/dev-build-and-up.sh` and `hack/build-go.sh`, you
can create a pod like this:


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
cluster/aws/kube-down.sh
```