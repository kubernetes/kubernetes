## Getting started on Google Compute Engine

### Prerequisites

1. You need a Google Cloud Platform account with billing enabled. Visit
   [http://cloud.google.com/console](http://cloud.google.com/console) for more details.
2. Make sure you can start up a GCE VM.  At least make sure you can do the [Create an instance](https://developers.google.com/compute/docs/quickstart#addvm) part of the GCE Quickstart.
3. You need to have the Google Storage API, and the Google Storage JSON API enabled.
4. You must have Go (version 1.2 or later) installed: [www.golang.org](http://www.golang.org).
5. You must have the [`gcloud` components](https://developers.google.com/cloud/sdk/) installed.
6. Ensure that your `gcloud` components are up-to-date by running `gcloud components update`.
7. Install godep (optional, only required when modifying package dependencies). [Instructions here](https://github.com/GoogleCloudPlatform/kubernetes#installing-godep)
8. Get the Kubernetes source:

        git clone https://github.com/GoogleCloudPlatform/kubernetes.git

### Setup

The setup script builds Kubernetes, then creates Google Compute Engine instances, firewall rules, and routes:

```
cd kubernetes
hack/dev-build-and-up.sh
```

The script above relies on Google Storage to deploy the software to instances running in GCE. It uses the Google Storage APIs so the "Google Cloud Storage JSON API" setting must be enabled for the project in the Google Developers Console (https://cloud.google.com/console#/project).

The instances must also be able to connect to each other using their private IP. The script uses the "default" network which should have a firewall rule called "default-allow-internal" which allows traffic on any port on the private IPs.
If this rule is missing from the default network or if you change the network being used in `cluster/config-default.sh` create a new rule with the following field values:
* Source Ranges: 10.0.0.0/8
* Allowed Protocols or Port: tcp:1-65535;udp:1-65535;icmp

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
cluster/kube-down.sh
```
