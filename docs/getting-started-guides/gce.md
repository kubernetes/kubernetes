<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
Getting started on Google Compute Engine
----------------------------------------

**Table of Contents**

- [Before you start](#before-you-start)
- [Prerequisites](#prerequisites)
- [Starting a cluster](#starting-a-cluster)
- [Installing the Kubernetes command line tools on your workstation](#installing-the-kubernetes-command-line-tools-on-your-workstation)
- [Getting started with your cluster](#getting-started-with-your-cluster)
    - [Inspect your cluster](#inspect-your-cluster)
    - [Run some examples](#run-some-examples)
- [Tearing down the cluster](#tearing-down-the-cluster)
- [Customizing](#customizing)
- [Troubleshooting](#troubleshooting)
    - [Project settings](#project-settings)
    - [Cluster initialization hang](#cluster-initialization-hang)
    - [SSH](#ssh)
    - [Networking](#networking)


The example below creates a Kubernetes cluster with 4 worker node Virtual Machines and a master Virtual Machine (i.e. 5 VMs in your cluster). This cluster is set up and controlled from your workstation (or wherever you find convenient).

### Before you start

If you want a simplified getting started experience and GUI for managing clusters, please consider trying [Google Container Engine](https://cloud.google.com/container-engine/) (GKE) for hosted cluster installation and management.

If you want to use custom binaries or pure open source Kubernetes, please continue with the instructions below.

### Prerequisites

1. You need a Google Cloud Platform account with billing enabled. Visit the [Google Developers Console](http://cloud.google.com/console) for more details.
1. Install `gcloud` as necessary. `gcloud` can be installed as a part of the [Google Cloud SDK](https://cloud.google.com/sdk/).
1. Then, make sure you have the `gcloud preview` command line component installed. Run `gcloud preview` at the command line - if it asks to install any components, go ahead and install them. If it simply shows help text, you're good to go. This is required as the cluster setup script uses GCE [Instance Groups](https://cloud.google.com/compute/docs/instance-groups/), which are in the gcloud preview namespace. You will also need to **enable [`Compute Engine Instance Group Manager API`](https://developers.google.com/console/help/new/#activatingapis)** in the developers console.
1. Make sure that gcloud is set to use the Google Cloud Platform project you want. You can check the current project using `gcloud config list project` and change it via `gcloud config set project <project-id>`.
1. Make sure you have credentials for GCloud by running ` gcloud auth login`.
1. Make sure you can start up a GCE VM from the command line.  At least make sure you can do the [Create an instance](https://cloud.google.com/compute/docs/instances/#startinstancegcloud) part of the GCE Quickstart.
1. Make sure you can ssh into the VM without interactive prompts.  See the [Log in to the instance](https://cloud.google.com/compute/docs/instances/#sshing) part of the GCE Quickstart.

### Starting a cluster

You can install a client and start a cluster with either one of these commands (we list both in case only one is installed on your machine):


 ```bash
 curl -sS https://get.k8s.io | bash
 ```

or

```bash
wget -q -O - https://get.k8s.io | bash
```

Once this command completes, you will have a master VM and four worker VMs, running as a Kubernetes cluster.

By default, some containers will already be running on your cluster. Containers like `kibana` and `elasticsearch` provide [logging](logging.md), while `heapster` provides [monitoring](http://releases.k8s.io/release-1.1/cluster/addons/cluster-monitoring/README.md) services.

The script run by the commands above creates a cluster with the name/prefix "kubernetes". It defines one specific cluster config, so you can't run it more than once.

Alternately, you can download and install the latest Kubernetes release from [this page](https://github.com/kubernetes/kubernetes/releases), then run the `<kubernetes>/cluster/kube-up.sh` script to start the cluster:

```bash
cd kubernetes
cluster/kube-up.sh
```

If you want more than one cluster running in your project, want to use a different name, or want a different number of worker nodes, see the `<kubernetes>/cluster/gce/config-default.sh` file for more fine-grained configuration before you start up your cluster.

If you run into trouble, please see the section on [troubleshooting](gce.md#troubleshooting), post to the
[google-containers group](https://groups.google.com/forum/#!forum/google-containers), or come ask questions on [Slack](../troubleshooting.md#slack).

The next few steps will show you:

1. how to set up the command line client on your workstation to manage the cluster
1. examples of how to use the cluster
1. how to delete the cluster
1. how to start clusters with non-default options (like larger clusters)

### Installing the Kubernetes command line tools on your workstation

The cluster startup script will leave you with a running cluster and a `kubernetes` directory on your workstation.
The next step is to make sure the `kubectl` tool is in your path.

The [kubectl](../user-guide/kubectl/kubectl.md) tool controls the Kubernetes cluster manager.  It lets you inspect your cluster resources, create, delete, and update components, and much more.
You will use it to look at your new cluster and bring up example apps.

Add the appropriate binary folder to your `PATH` to access kubectl:

```bash
# OS X
export PATH=<path/to/kubernetes-directory>/platforms/darwin/amd64:$PATH

# Linux
export PATH=<path/to/kubernetes-directory>/platforms/linux/amd64:$PATH
```

**Note**: gcloud also ships with `kubectl`, which by default is added to your path.
However the gcloud bundled kubectl version may be older than the one downloaded by the
get.k8s.io install script. We recommend you use the downloaded binary to avoid
potential issues with client/server version skew.

#### Enabling bash completion of the Kubernetes command line tools

You may find it useful to enable `kubectl` bash completion:

```
$ source ./contrib/completions/bash/kubectl
```

**Note**: This will last for the duration of your bash session. If you want to make this permanent you need to add this line in your bash profile.

Alternatively, on most linux distributions you can also move the completions file to your bash_completions.d like this:

```
$ cp ./contrib/completions/bash/kubectl /etc/bash_completion.d/
```

but then you have to update it when you update kubectl.

### Getting started with your cluster

#### Inspect your cluster

Once `kubectl` is in your path, you can use it to look at your cluster. E.g., running:

```console
$ kubectl get --all-namespaces services
```

should show a set of [services](../user-guide/services.md) that look something like this:

```console
NAMESPACE     NAME                  CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
default       kubernetes            10.0.0.1         <none>            443/TCP       <none>                 1d
kube-system   kube-dns              10.0.0.2         <none>            53/TCP,53/UDP k8s-app=kube-dns       1d
kube-system   kube-ui               10.0.0.3         <none>            80/TCP        k8s-app=kube-ui        1d
...
```

Similarly, you can take a look at the set of [pods](../user-guide/pods.md) that were created during cluster startup.
You can do this via the

```console
$ kubectl get --all-namespaces pods
```

command.

You'll see a list of pods that looks something like this (the name specifics will be different):

```console
NAMESPACE     NAME                                           READY     STATUS    RESTARTS   AGE
kube-system   fluentd-cloud-logging-kubernetes-minion-63uo   1/1       Running   0          14m
kube-system   fluentd-cloud-logging-kubernetes-minion-c1n9   1/1       Running   0          14m
kube-system   fluentd-cloud-logging-kubernetes-minion-c4og   1/1       Running   0          14m
kube-system   fluentd-cloud-logging-kubernetes-minion-ngua   1/1       Running   0          14m
kube-system   kube-dns-v5-7ztia                              3/3       Running   0          15m
kube-system   kube-ui-v1-curt1                               1/1       Running   0          15m
kube-system   monitoring-heapster-v5-ex4u3                   1/1       Running   1          15m
kube-system   monitoring-influx-grafana-v1-piled             2/2       Running   0          15m
```

Some of the pods may take a few seconds to start up (during this time they'll show `Pending`), but check that they all show as `Running` after a short period.

#### Run some examples

Then, see [a simple nginx example](../../docs/user-guide/simple-nginx.md) to try out your new cluster.

For more complete applications, please look in the [examples directory](../../examples/).  The [guestbook example](../../examples/guestbook/) is a good "getting started" walkthrough.

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

Also ensure that-- as listed in the [Prerequsites section](#prerequisites)-- you've enabled the `Compute Engine Instance Group Manager API`, and can start up a GCE VM from the command line as in the [GCE Quickstart](https://cloud.google.com/compute/docs/quickstart) instructions.

#### Cluster initialization hang

If the Kubernetes startup script hangs waiting for the API to be reachable, you can troubleshoot by SSHing into the master and node VMs and looking at logs such as `/var/log/startupscript.log`.

**Once you fix the issue, you should run `kube-down.sh` to cleanup** after the partial cluster creation, before running `kube-up.sh` to try again.

#### SSH

If you're having trouble SSHing into your instances, ensure the GCE firewall
isn't blocking port 22 to your VMs.  By default, this should work but if you
have edited firewall rules or created a new non-default network, you'll need to
expose it: `gcloud compute firewall-rules create default-ssh --network=<network-name>
--description "SSH allowed from anywhere" --allow tcp:22`

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




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/gce.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
