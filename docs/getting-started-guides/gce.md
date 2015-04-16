## Getting started on Google Compute Engine

The example below creates a Kubernetes cluster with 4 worker node Virtual Machines and a master Virtual Machine (i.e. 5 VMs in your cluster). This cluster is set up and controlled from your workstation (or wherever you find convenient).

### Before you start

If you want a simplified getting started experience and GUI for managing clusters, please consider trying [Google Container Engine](https://cloud.google.com/container-engine/) for hosted cluster installation and management.  

If you want to use custom binaries or pure open source Kubernetes, please continue with the instructions below.

### Prerequisites

1. You need a Google Cloud Platform account with billing enabled. Visit the [Google Developers Console](http://cloud.google.com/console) for more details.
1. Make sure you can start up a GCE VM from the command line.  At least make sure you can do the [Create an instance](https://cloud.google.com/compute/docs/quickstart#create_an_instance) part of the GCE Quickstart.
1. Make sure you have the `gcloud preview` command line component installed. Simply run `gcloud preview` at the command line - if it asks to install any components, go ahead and install them. If it simply shows help text, you're good to go.
1. Make sure you can ssh into the VM without interactive prompts.  See the [Log in to the instance](https://cloud.google.com/compute/docs/quickstart#ssh) part of the GCE Quickstart.

### Starting a Cluster

You can install a cluster with one of two one-liners:

```bash
curl -sS https://get.k8s.io | bash
```

or

```bash
wget -q -O - https://get.k8s.io | bash
```

### Installing the kubernetes client on your workstation

This will leave you with a ```kubernetes``` directory on your workstation, and a running cluster.

Feel free to move the ```kubernetes``` directory to the appropriate directory on your workstation (e.g. ```/opt/kubernetes```) then ```cd``` into that directory:

```bash
mv kubernetes ${SOME_DIR}/kubernetes
cd ${SOME_DIR}/kubernetes
```

If you run into trouble please see the section on [troubleshooting](https://github.com/brendandburns/kubernetes/blob/docs/docs/getting-started-guides/gce.md#troubleshooting), or come ask questions on IRC at #google-containers on freenode.


### Running a container (simple version)

Once you have your cluster created you can use ```${SOME_DIR}/kubernetes/cluster/kubectl.sh``` to access
the kubernetes api.

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

```bash
cd kubernetes
cluster/kubectl.sh create -f docs/getting-started-guides/pod.json
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
        "image": "nginx",
        "ports": [{
          "containerPort": 80,
          "hostPort": 8081
        }],
        "livenessProbe": {
          "enabled": true,
          "type": "http",
          "initialDelaySeconds": 30,
          "httpGet": {
            "path": "/index.html",
            "port": 8081
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
cluster/kubectl.sh get pods
```

and delete the pod you just created:

```bash
cluster/kubectl.sh delete pods php
```

Since this pod is scheduled on a minion running in GCE, you will have to enable incoming tcp traffic via the port specified in the
pod manifest before you see the nginx welcome page. After doing so, it should be visible at http://<external ip of minion running nginx>:<port from manifest>.

Look in `examples/` for more examples

### Tearing down the cluster

```bash
cd kubernetes
cluster/kube-down.sh
```

### Customizing

The script above relies on Google Storage to stage the Kubernetes release. It
then will start (by default) a single master VM along with 4 worker VMs.  You
can tweak some of these parameters by editing `kubernetes/cluster/gce/config-default.sh`
You can view a transcript of a successful cluster creation
[here](https://gist.github.com/satnam6502/fc689d1b46db9772adea).

### Troubleshooting

#### Project settings

You need to have the Google Cloud Storage API, and the Google Cloud Storage
JSON API enabled. It is activated by default for new projects. Otherwise, it
can be done in the Google Cloud Console.  See the [Google Cloud Storage JSON
API Overview](https://cloud.google.com/storage/docs/json_api/) for more
details.

#### SSH

If you're having trouble SSHing into your instances, ensure the GCE firewall
isn't blocking port 22 to your VMs.  By default, this should work but if you
have edited firewall rules or created a new non-default network, you'll need to
expose it: `gcloud compute firewall-rules create --network=<network-name>
--description "SSH allowed from anywhere" --allow tcp:22 default-ssh`

Additionally, your GCE SSH key must either have no passcode or you need to be
using `ssh-agent`.

#### Networking

The instances must be able to connect to each other using their private IP. The
script uses the "default" network which should have a firewall rule called
"default-allow-internal" which allows traffic on any port on the private IPs.
If this rule is missing from the default network or if you change the network
being used in `cluster/config-default.sh` create a new rule with the following
field values:

* Source Ranges: `10.0.0.0/8`
* Allowed Protocols and Port: `tcp:1-65535;udp:1-65535;icmp`
