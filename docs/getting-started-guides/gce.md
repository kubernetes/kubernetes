## Getting started on Google Compute Engine

The example below creates a Kubernetes cluster with 4 worker node Virtual Machines and a master Virtual Machine (i.e. 5 VMs in your cluster). This cluster is set up and controlled from your workstation (or wherever you find convenient).

### Getting VMs

1. You need a Google Cloud Platform account with billing enabled. Visit
   [http://cloud.google.com/console](http://cloud.google.com/console) for more details.
2. Make sure you can start up a GCE VM.  At least make sure you can do the [Create an instance](https://developers.google.com/compute/docs/quickstart#addvm) part of the GCE Quickstart.
3. Make sure you can ssh into the VM without interactive prompts.
  * Your GCE SSH key must either have no passcode or you need to be using `ssh-agent`.
  * Ensure the GCE firewall isn't blocking port 22 to your VMs.  By default, this should work but if you have edited firewall rules or created a new non-default network, you'll need to expose it: `gcutil addfirewall --network=<network-name> --description "SSH allowed from anywhere" --allowed=tcp:22 default-ssh`
4. You need to have the Google Cloud Storage API, and the Google Cloud Storage JSON API enabled. This can be done in the Google Cloud Console.


### Prerequisites for your workstation

1. Be running a Linux or Mac OS X.
2. You must have the [Google Cloud SDK](https://developers.google.com/cloud/sdk/) installed.  This will get you `gcloud`, `gcutil` and `gsutil`.
3. Ensure that your `gcloud` components are up-to-date by running `gcloud components update`.
4. If you want to build your own release, you need to have [Docker
installed](https://docs.docker.com/installation/).  On Mac OS X you can use
boot2docker.
5. Get or build a [binary release](binary_release.md)

### Starting a Cluster

```bash
cluster/kube-up.sh
```

The script above relies on Google Storage to stage the Kubernetes release. It
then will start (by default) a single master VM along with 4 worker VMs.  You
can tweak some of these parameters by editing `cluster/gce/config-default.sh`
You can view a transcript of a successful cluster creation
[here](https://gist.github.com/satnam6502/fc689d1b46db9772adea).

The instances must be able to connect to each other using their private IP. The
script uses the "default" network which should have a firewall rule called
"default-allow-internal" which allows traffic on any port on the private IPs.
If this rule is missing from the default network or if you change the network
being used in `cluster/config-default.sh` create a new rule with the following
field values:

* Source Ranges: `10.0.0.0/8`
* Allowed Protocols and Port: `tcp:1-65535;udp:1-65535;icmp`

### Running a container (simple version)

Once you have your instances up and running, the `build-go.sh` script sets up
your Go workspace and builds the Go components.

The `kubecfg.sh` line below spins up two containers running
[Nginx](http://nginx.org/en/) with port 80 mapped to 8080:

```bash
cluster/kubecfg.sh -p 8080:80 run dockerfile/nginx 2 myNginx
```

To stop the containers:

```bash
cluster/kubecfg.sh stop myNginx
```

To delete the containers:

```bash
cluster/kubecfg.sh rm myNginx
```

### Running a container (more complete version)


Assuming you've run `hack/dev-build-and-up.sh` and `hack/build-go.sh`, you
can create a pod like this:


```bash
cd kubernetes
cluster/kubecfg.sh -c api/examples/pod.json create /pods
```

Where pod.json contains something like:

```json
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

```bash
cluster/kubecfg.sh list pods
```

and delete the pod you just created:

```bash
cluster/kubecfg.sh delete pods/php
```

Look in `examples/` for more examples

### Tearing down the cluster
```bash
cd kubernetes
cluster/kube-down.sh
```
