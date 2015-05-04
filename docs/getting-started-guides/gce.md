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

Copy the appropriate ```kubectl``` binary to somewhere in your ```PATH```, for example:

```bash
# OS X
sudo cp kubernetes/platforms/darwin/amd64/kubectl /usr/local/bin/kubectl

# Linux
sudo cp kubernetes/platforms/linux/amd64/kubectl /usr/local/bin/kubectl
```

If you run into trouble please see the section on [troubleshooting](https://github.com/brendandburns/kubernetes/blob/docs/docs/getting-started-guides/gce.md#troubleshooting), or come ask questions on IRC at #google-containers on freenode.


### Getting started with your cluster
See [a simple nginx example](../../examples/simple-nginx.md) to try out your new cluster.

For more complete applications, please look in the [examples directory](../../examples)


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
