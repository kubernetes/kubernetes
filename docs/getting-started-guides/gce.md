## Getting started on Google Compute Engine

The example below creates a Kubernetes cluster with 4 worker node Virtual Machines and a master Virtual Machine (i.e. 5 VMs in your cluster). This cluster is set up and controlled from your workstation (or wherever you find convenient).

### Before you start

If you want a simplified getting started experience and GUI for managing clusters, please consider trying [Google Container Engine](https://cloud.google.com/container-engine/) for hosted cluster installation and management.

If you want to use custom binaries or pure open source Kubernetes, please continue with the instructions below.

### Prerequisites

1. You need a Google Cloud Platform account with billing enabled. Visit the [Google Developers Console](http://cloud.google.com/console) for more details.
1. Make sure you have the `gcloud preview` command line component installed. Simply run `gcloud preview` at the command line - if it asks to install any components, go ahead and install them. If it simply shows help text, you're good to go. This is required as the cluster setup script uses GCE [Instance Groups](https://cloud.google.com/compute/docs/instance-groups/), which are in the gcloud preview namespace. You will also need to enable `Compute Engine Instance Group Manager API` in the developers console. `gcloud` can be installed as a part of the [Google Cloud SDK](https://cloud.google.com/sdk/)
1. Make sure that gcloud is set to use the Google Cloud Platform project you want. You can check the current project using `gcloud config list project` and change it via `gcloud config set project <project-id>`.
1. Make sure you have credentials for GCloud by running ` gcloud auth login`.
1. Make sure you can start up a GCE VM from the command line.  At least make sure you can do the [Create an instance](https://cloud.google.com/compute/docs/quickstart#create_an_instance) part of the GCE Quickstart.
1. Make sure you can ssh into the VM without interactive prompts.  See the [Log in to the instance](https://cloud.google.com/compute/docs/quickstart#ssh) part of the GCE Quickstart.

### Starting a Cluster

You can install a client and start a cluster with this command:

```bash
curl -sS https://get.k8s.io | bash
```

Once this command completes, you will have a master VM and four worker VMs, running as a Kubernetes cluster. By default, some containers will already be running on your cluster. Containers like `kibana` and `elasticsearch` provide [logging](../logging.md), while `heapster` provides [monitoring](../../cluster/addons/cluster-monitoring/README.md) services.

If you run into trouble please see the section on [troubleshooting](gce.md#troubleshooting), or come ask questions on IRC at #google-containers on freenode.

The next few steps will show you:

1. how to set up the command line client on your workstation to manage the cluster
1. examples of how to use the cluster
1. how to delete the cluster
1. how to start clusters with non-default options (like larger clusters)

### Installing the kubernetes command line tools on your workstation

The cluster startup script will leave you with a running cluster and a ```kubernetes``` directory on your workstation.

Add the appropriate binary folder to your ```PATH``` to access kubectl:

```bash
# OS X
export PATH=<path/to/kubernetes-directory>/platforms/darwin/amd64:$PATH

# Linux
export PATH=<path/to/kubernetes-directory>/platforms/linux/amd64:$PATH
```

Note: gcloud also ships with ```kubectl```, which by default is added to your path.
However the gcloud bundled kubectl version may be older than the one downloaded by the
get.k8s.io install script. We recommend you use the downloaded binary to avoid
potential issues with client/server version skew.

### Getting started with your cluster
See [a simple nginx example](../../examples/simple-nginx.md) to try out your new cluster.

For more complete applications, please look in the [examples directory](../../examples).

### Tearing down the cluster
To remove/delete/teardown the cluster, use the `kube-down.sh` script.

```bash
cd kubernetes
cluster/kube-down.sh
```

Likewise, the `kube-up.sh` in the same directory will bring it back up. You do not need to rerun the `curl` or `wget` command: everything needed to setup the Kubernetes cluster is now on your workstation.

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

#### Cluster initialization hang

If the Kubernetes startup script hangs waiting for the API to be reachable, you can troubleshoot by SSHing into the master and minion VMs and looking at logs such as `/var/log/startupscript.log`.

Once you fix the issue, you should run `kube-down.sh` to cleanup after the partial cluster creation, before running `kube-up.sh` to try again.

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


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/gce.md?pixel)]()
