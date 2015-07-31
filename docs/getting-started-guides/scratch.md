<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/scratch.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Getting started from Scratch
----------------------------

This guide is for people who want to craft a custom Kubernetes cluster.  If you
can find an existing Getting Started Guide that meets your needs on [this
list](README.md), then we recommend using it, as you will be able to benefit
from the experience of others.  However, if you have specific IaaS, networking,
configuration management, or operating system requirements not met by any of
those guides, then this guide will provide an outline of the steps you need to
take.  Note that it requires considerably more effort than using one of the
pre-defined guides.

This guide is also useful for those wanting to understand at a high level some of the
steps that existing cluster setup scripts are making.

**Table of Contents**

<!-- BEGIN MUNGE: GENERATED_TOC -->

  - [Designing and Preparing](#designing-and-preparing)
    - [Learning](#learning)
    - [Cloud Provider](#cloud-provider)
    - [Nodes](#nodes)
    - [Network](#network)
    - [Cluster Naming](#cluster-naming)
    - [Software Binaries](#software-binaries)
      - [Downloading and Extracting Kubernetes Binaries](#downloading-and-extracting-kubernetes-binaries)
      - [Selecting Images](#selecting-images)
    - [Security Models](#security-models)
      - [Preparing Certs](#preparing-certs)
      - [Preparing Credentials](#preparing-credentials)
  - [Configuring and Installing Base Software on Nodes](#configuring-and-installing-base-software-on-nodes)
    - [Docker](#docker)
    - [rkt](#rkt)
    - [kubelet](#kubelet)
    - [kube-proxy](#kube-proxy)
    - [Networking](#networking)
    - [Other](#other)
    - [Using Configuration Management](#using-configuration-management)
  - [Bootstrapping the Cluster](#bootstrapping-the-cluster)
    - [etcd](#etcd)
    - [Apiserver, Controller Manager, and Scheduler](#apiserver-controller-manager-and-scheduler)
      - [Apiserver pod template](#apiserver-pod-template)
        - [Cloud Providers](#cloud-providers)
      - [Scheduler pod template](#scheduler-pod-template)
      - [Controller Manager Template](#controller-manager-template)
      - [Starting and Verifying Apiserver, Scheduler, and Controller Manager](#starting-and-verifying-apiserver-scheduler-and-controller-manager)
    - [Logging](#logging)
    - [Monitoring](#monitoring)
    - [DNS](#dns)
  - [Troubleshooting](#troubleshooting)
    - [Running validate-cluster](#running-validate-cluster)
    - [Inspect pods and services](#inspect-pods-and-services)
    - [Try Examples](#try-examples)
    - [Running the Conformance Test](#running-the-conformance-test)
    - [Networking](#networking)
    - [Getting Help](#getting-help)

<!-- END MUNGE: GENERATED_TOC -->

## Designing and Preparing

### Learning

  1. You should be familiar with using Kubernetes already.  We suggest you set
    up a temporary cluster by following one of the other Getting Started Guides.
    This will help you become familiar with the CLI ([kubectl](../user-guide/kubectl/kubectl.md)) and concepts ([pods](../user-guide/pods.md), [services](../user-guide/services.md), etc.) first.
  1. You should have `kubectl` installed on your desktop.  This will happen as a side
    effect of completing one of the other Getting Started Guides.  If not, follow the instructions
    [here](../user-guide/prereqs.md).

### Cloud Provider

Kubernetes has the concept of a Cloud Provider, which is a module which provides
an interface for managing TCP Load Balancers, Nodes (Instances) and Networking Routes.
The interface is defined in `pkg/cloudprovider/cloud.go`.  It is possible to
create a custom cluster without implementing a cloud provider (for example if using
bare-metal), and not all parts of the interface need to be implemented, depending
on how flags are set on various components.

### Nodes

- You can use virtual or physical machines.
- While you can build a cluster with 1 machine, in order to run all the examples and tests you
  need at least 4 nodes.
- Many Getting-started-guides make a distinction between the master node and regular nodes.  This
  is not strictly necessary.
- Nodes will need to run some version of Linux with the x86_64 architecture.  It may be possible
  to run on other OSes and Architectures, but this guide does not try to assist with that.
- Apiserver and etcd together are fine on a machine with 1 core and 1GB RAM for clusters with 10s of nodes.
  Larger or more active clusters may benefit from more cores.
- Other nodes can have any reasonable amount of memory and any number of cores.  They need not
  have identical configurations.

### Network

Kubernetes has a distinctive [networking model](../admin/networking.md).

Kubernetes allocates an IP address to each pod.  When creating a cluster, you
need to allocate a block of IPs for Kubernetes to use as Pod IPs.  The simplest
approach is to allocate a different block of IPs to each node in the cluster as
the node is added.  A process in one pod should be able to communicate with
another pod using the IP of the second pod.  This connectivity can be
accomplished in two ways:
- Configure network to route Pod IPs
  - Harder to setup from scratch.
  - Google Compute Engine ([GCE](gce.md)) and [AWS](aws.md) guides use this approach.
  - Need to make the Pod IPs routable by programming routers, switches, etc.
  - Can be configured external to Kubernetes, or can implement in the "Routes" interface of a Cloud Provider module.
  - Generally highest performance.
- Create an Overlay network
  - Easier to setup
  - Traffic is encapsulated, so per-pod IPs are routable.
  - Examples:
    - [Flannel](https://github.com/coreos/flannel)
    - [Weave](http://weave.works/)
    - [Open vSwitch (OVS)](http://openvswitch.org/)
  - Does not require "Routes" portion of Cloud Provider module.
  - Reduced performance (exactly how much depends on your solution).

You need to select an address range for the Pod IPs.
- Various approaches:
  - GCE: each project has its own `10.0.0.0/8`.  Carve off a `/16` for each
    Kubernetes cluster from that space, which leaves room for several clusters.
    Each node gets a further subdivision of this space.
  - AWS: use one VPC for whole organization, carve off a chunk for each
    cluster, or use different VPC for different clusters.
  - IPv6 is not supported yet.
- Allocate one CIDR subnet for each node's PodIPs, or a single large CIDR
  from which smaller CIDRs are automatically allocated to each node (if nodes
  are dynamically added).
  - You need max-pods-per-node * max-number-of-nodes IPs in total. A `/24` per
    node supports 254 pods per machine and is a common choice.  If IPs are
    scarce, a `/26` (62 pods per machine) or even a `/27` (30 pods) may be sufficient.
  - e.g. use `10.10.0.0/16` as the range for the cluster, with up to 256 nodes
    using `10.10.0.0/24` through `10.10.255.0/24`, respectively.
  - Need to make these routable or connect with overlay.

Kubernetes also allocates an IP to each [service](../user-guide/services.md).  However,
service IPs do not necessarily need to be routable.  The kube-proxy takes care
of translating Service IPs to Pod IPs before traffic leaves the node.  You do
need to Allocate a block of IPs for services.  Call this
`SERVICE_CLUSTER_IP_RANGE`.  For example, you could set
`SERVICE_CLUSTER_IP_RANGE="10.0.0.0/16"`, allowing 65534 distinct services to
be active at once.  Note that you can grow the end of this range, but you
cannot move it without disrupting the services and pods that already use it.

Also, you need to pick a static IP for master node.
- Call this `MASTER_IP`.
- Open any firewalls to allow access to the apiserver ports 80 and/or 443.
- Enable ipv4 forwarding sysctl, `net.ipv4.ip_forward = 1`

### Cluster Naming

You should pick a name for your cluster.  Pick a short name for each cluster
which is unique from future cluster names. This will be used in several ways:
  - by kubectl to distinguish between various clusters you have access to.  You will probably want a
    second one sometime later, such as for testing new Kubernetes releases, running in a different
region of the world, etc.
  - Kubernetes clusters can create cloud provider resources (e.g. AWS ELBs) and different clusters
    need to distinguish which resources each created.  Call this `CLUSTERNAME`.

### Software Binaries

You will need binaries for:
  - etcd
  - A container runner, one of:
    - docker
    - rkt
  - Kubernetes
    - kubelet
    - kube-proxy
    - kube-apiserver
    - kube-controller-manager
    - kube-scheduler

#### Downloading and Extracting Kubernetes Binaries

A Kubernetes binary release includes all the Kubernetes binaries as well as the supported release of etcd.
You can use a Kubernetes binary release (recommended) or build your Kubernetes binaries following the instructions in the
[Developer Documentation](../devel/README.md).  Only using a binary release is covered in this guide.

Download the [latest binary release](https://github.com/GoogleCloudPlatform/kubernetes/releases/latest) and unzip it.
Then locate `./kubernetes/server/kubernetes-server-linux-amd64.tar.gz` and unzip *that*.
Then, within the second set of unzipped files, locate `./kubernetes/server/bin`, which contains
all the necessary binaries.

#### Selecting Images

You will run docker, kubelet, and kube-proxy outside of a container, the same way you would run any system daemon, so
you just need the bare binaries.  For etcd, kube-apiserver, kube-controller-manager, and kube-scheduler,
we recommend that you run these as containers, so you need an image to be built.

You have several choices for Kubernetes images:
- Use images hosted on Google Container Registry (GCR):
  - e.g `gcr.io/google_containers/kube-apiserver:$TAG`, where `TAG` is the latest
    release tag, which can be found on the [latest releases page](https://github.com/GoogleCloudPlatform/kubernetes/releases/latest).
  - Ensure $TAG is the same tag as the release tag you are using for kubelet and kube-proxy.
- Build your own images.
  - Useful if you are using a private registry.
  - The release contains files such as `./kubernetes/server/bin/kube-apiserver.tar` which
    can be converted into docker images using a command like
    `docker load -i kube-apiserver.tar`
  - You can verify if the image is loaded successfully with the right reposity and tag using
    command like `docker images`

For etcd, you can:
- Use images hosted on Google Container Registry (GCR), such as `gcr.io/google_containers/etcd:2.0.12`
- Use images hosted on [Docker Hub](https://registry.hub.docker.com/u/coreos/etcd/) or [quay.io](https://registry.hub.docker.com/u/coreos/etcd/)
- Use etcd binary included in your OS distro.
- Build your own image
  - You can do: `cd kubernetes/cluster/images/etcd; make`

We recommend that you use the etcd version which is provided in the Kubernetes binary distribution.   The Kubernetes binaries in the release
were tested extensively with this version of etcd and not with any other version.
The recommended version number can also be found as the value of `ETCD_VERSION` in `kubernetes/cluster/images/etcd/Makefile`.

The remainder of the document assumes that the image identifiers have been chosen and stored in corresponding env vars.  Examples (replace with latest tags and appropriate registry):
  - `APISERVER_IMAGE=gcr.io/google_containers/kube-apiserver:$TAG`
  - `SCHEDULER_IMAGE=gcr.io/google_containers/kube-scheduler:$TAG`
  - `CNTRLMNGR_IMAGE=gcr.io/google_containers/kube-controller-manager:$TAG`
  - `ETCD_IMAGE=gcr.io/google_containers/etcd:$ETCD_VERSION`

### Security Models

There are two main options for security:
- Access the apiserver using HTTP.
  - Use a firewall for security.
  - This is easier to setup.
- Access the apiserver using HTTPS
  - Use https with certs, and credentials for user.
  - This is the recommended approach.
  - Configuring certs can be tricky.

If following the HTTPS approach, you will need to prepare certs and credentials.

#### Preparing Certs

You need to prepare several certs:
- The master needs a cert to act as an HTTPS server.
- The kubelets optionally need certs to identify themselves as clients of the master, and when
  serving its own API over HTTPS.

Unless you plan to have a real CA generate your certs, you will need to generate a root cert and use that to sign the master, kubelet, and kubectl certs.
- see function `create-certs` in `cluster/gce/util.sh`
- see also `cluster/saltbase/salt/generate-cert/make-ca-cert.sh` and
  `cluster/saltbase/salt/generate-cert/make-cert.sh`

You will end up with the following files (we will use these variables later on)
- `CA_CERT`
  - put in on node where apiserver runs, in e.g. `/srv/kubernetes/ca.crt`.
- `MASTER_CERT`
  - signed by CA_CERT
  - put in on node where apiserver runs, in e.g. `/srv/kubernetes/server.crt`
- `MASTER_KEY `
  - put in on node where apiserver runs, in e.g. `/srv/kubernetes/server.key`
- `KUBELET_CERT`
  - optional
- `KUBELET_KEY`
  - optional

#### Preparing Credentials

The admin user (and any users) need:
  - a token or a password to identify them.
  - tokens are just long alphanumeric strings, e.g. 32 chars.  See
    - `TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)`

Your tokens and passwords need to be stored in a file for the apiserver
to read.  This guide uses `/var/lib/kube-apiserver/known_tokens.csv`.
The format for this file is described in the [authentication documentation](../admin/authentication.md).

For distributing credentials to clients, the convention in Kubernetes is to put the credentials
into a [kubeconfig file](../user-guide/kubeconfig-file.md).

The kubeconfig file for the administrator can be created as follows:
 - If you have already used Kubernetes with a non-custom cluster (for example, used a Getting Started
   Guide), you will already have a `$HOME/.kube/config` file.
 - You need to add certs, keys, and the master IP to the kubeconfig file:
    - If using the firewall-only security option, set the apiserver this way:
      - `kubectl config set-cluster $CLUSTER_NAME --server=http://$MASTER_IP --insecure-skip-tls-verify=true`
    - Otherwise, do this to set the apiserver ip, client certs, and user credentials.
      - `kubectl config set-cluster $CLUSTER_NAME --certificate-authority=$CA_CERT --embed-certs=true --server=https://$MASTER_IP`
      - `kubectl config set-credentials $USER --client-certificate=$CLI_CERT --client-key=$CLI_KEY --embed-certs=true --token=$TOKEN`
    - Set your cluster as the default cluster to use:
      - `kubectl config set-context $CONTEXT_NAME --cluster=$CLUSTER_NAME --user=$USER`
      - `kubectl config use-context $CONTEXT_NAME`

Next, make a kubeconfig file for the kubelets and kube-proxy.  There are a couple of options for how
many distinct files to make:
  1. Use the same credential as the admin
    - This is simplest to setup.
  1. One token and kubeconfig file for all kubelets, one for all kube-proxy, one for admin.
    - This mirrors what is done on GCE today
  1. Different credentials for every kubelet, etc.
    - We are working on this but all the pieces are not ready yet.

You can make the files by copying the `$HOME/.kube/config`, by following the code
in `cluster/gce/configure-vm.sh` or by using the following template:

```yaml
apiVersion: v1
kind: Config
users:
- name: kubelet
  user:
    token: ${KUBELET_TOKEN}
clusters:
- name: local
  cluster:
    certificate-authority-data: ${CA_CERT_BASE64_ENCODED}
contexts:
- context:
    cluster: local
    user: kubelet
  name: service-account-context
current-context: service-account-context
```

Put the kubeconfig(s) on every node.  The examples later in this
guide assume that there are kubeconfigs in `/var/lib/kube-proxy/kubeconfig` and
`/var/lib/kubelet/kubeconfig`.

## Configuring and Installing Base Software on Nodes

This section discusses how to configure machines to be Kubernetes nodes.

You should run three daemons on every node:
  - docker or rkt
  - kubelet
  - kube-proxy

You will also need to do assorted other configuration on top of a
base OS install.

Tip: One possible starting point is to setup a cluster using an existing Getting
Started Guide.   After getting a cluster running, you can then copy the init.d scripts or systemd unit files from that
cluster, and then modify them for use on your custom cluster.

### Docker

The minimum required Docker version will vary as the kubelet version changes.  The newest stable release is a good choice.  Kubelet will log a warning and refuse to start pods if the version is too old, so pick a version and try it.

If you previously had Docker installed on a node without setting Kubernetes-specific
options, you may have a Docker-created bridge and iptables rules.  You may want to remove these
as follows before proceeding to configure Docker for Kubernetes.

```sh
iptables -t nat -F 
ifconfig docker0 down
brctl delbr docker0
```

The way you configure docker will depend in whether you have chosen the routable-vip or overlay-network approaches for your network.
Some suggested docker options:
  - create your own bridge for the per-node CIDR ranges, call it cbr0, and set `--bridge=cbr0` option on docker.
  - set `--iptables=false` so docker will not manipulate iptables for host-ports (too coarse on older docker versions, may be fixed in newer versions)
so that kube-proxy can manage iptables instead of docker.
  - `--ip-masq=false`
    - if you have setup PodIPs to be routable, then you want this false, otherwise, docker will
      rewrite the PodIP source-address to a NodeIP.
    - some environments (e.g. GCE) still need you to masquerade out-bound traffic when it leaves the cloud environment. This is very environment specific.
    - if you are using an overlay network, consult those instructions.
  - `--mtu=`
    - may be required when using Flannel, because of the extra packet size due to udp encapsulation
  - `--insecure-registry $CLUSTER_SUBNET`
    - to connect to a private registry, if you set one up, without using SSL.

You may want to increase the number of open files for docker:
   - `DOCKER_NOFILE=1000000`

Where this config goes depends on your node OS.  For example, GCE's Debian-based distro uses `/etc/default/docker`.

Ensure docker is working correctly on your system before proceeding with the rest of the
installation, by following examples given in the Docker documentation.

### rkt

[rkt](https://github.com/coreos/rkt) is an alternative to Docker.  You only need to install one of Docker or rkt.
The minimum version required is [v0.5.6](https://github.com/coreos/rkt/releases/tag/v0.5.6).

[systemd](http://www.freedesktop.org/wiki/Software/systemd/) is required on your node to run rkt.  The
minimum version required to match rkt v0.5.6 is
[systemd 215](http://lists.freedesktop.org/archives/systemd-devel/2014-July/020903.html).

[rkt metadata service](https://github.com/coreos/rkt/blob/master/Documentation/networking.md) is also required
for rkt networking support.  You can start rkt metadata service by using command like
`sudo systemd-run rkt metadata-service`

Then you need to configure your kubelet with flag:
  - `--container_runtime=rkt`

### kubelet

All nodes should run kubelet.  See [Selecting Binaries](#selecting-binaries).

Arguments to consider:
  - If following the HTTPS security approach:
    - `--api-servers=https://$MASTER_IP`
    - `--kubeconfig=/var/lib/kubelet/kubeconfig`
  - Otherwise, if taking the firewall-based security approach
    - `--api-servers=http://$MASTER_IP`
  - `--config=/etc/kubernetes/manifests`
  - `--cluster-dns=` to the address of the DNS server you will setup (see [Starting Addons](#starting-addons).)
  - `--cluster-domain=` to the dns domain prefix to use for cluster DNS addresses.
  - `--docker-root=`
  - `--root-dir=`
  - `--configure-cbr0=` (described above)
  - `--register-node` (described in [Node](../admin/node.md) documentation.)

### kube-proxy

All nodes should run kube-proxy.  (Running kube-proxy on a "master" node is not
strictly required, but being consistent is easier.)   Obtain a binary as described for
kubelet.

Arguments to consider:
  - If following the HTTPS security approach:
    - `--api-servers=https://$MASTER_IP`
    - `--kubeconfig=/var/lib/kube-proxy/kubeconfig`
  - Otherwise, if taking the firewall-based security approach
    - `--api-servers=http://$MASTER_IP`

### Networking

Each node needs to be allocated its own CIDR range for pod networking.
Call this `NODE_X_POD_CIDR`.

A bridge called `cbr0` needs to be created on each node.  The bridge is explained
further in the [networking documentation](../admin/networking.md).  The bridge itself
needs an address from `$NODE_X_POD_CIDR` - by convention the first IP.  Call
this `NODE_X_BRIDGE_ADDR`.  For example, if `NODE_X_POD_CIDR` is `10.0.0.0/16`,
then `NODE_X_BRIDGE_ADDR` is `10.0.0.1/16`.  NOTE: this retains the `/16` suffix
because of how this is used later.

- Recommended, automatic approach:
  1. Set `--configure-cbr0=true` option in kubelet init script and restart kubelet service.  Kubelet will configure cbr0 automatically.
     It will wait to do this until the node controller has set Node.Spec.PodCIDR.  Since you have not setup apiserver and node controller
     yet, the bridge will not be setup immediately.
- Alternate, manual approach:
  1. Set `--configure-cbr0=false` on kubelet and restart.
  1. Create a bridge
  - e.g. `brctl addbr cbr0`.
  1. Set appropriate MTU
  - `ip link set dev cbr0 mtu 1460` (NOTE: the actual value of MTU will depend on your network environment)
  1. Add the clusters network to the bridge (docker will go on other side of bridge).
  - e.g. `ip addr add $NODE_X_BRIDGE_ADDR dev eth0`
  1. Turn it on
  - e.g. `ip link set dev cbr0 up`

If you have turned off Docker's IP masquerading to allow pods to talk to each
other, then you may need to do masquerading just for destination IPs outside
the cluster network.  For example:

```sh
iptables -w -t nat -A POSTROUTING -o eth0 -j MASQUERADE \! -d ${CLUSTER_SUBNET}
```

This will rewrite the source address from
the PodIP to the Node IP for traffic bound outside the cluster, and kernel
[connection tracking](http://www.iptables.info/en/connection-state.html)
will ensure that responses destined to the node still reach
the pod.

NOTE: This is environment specific.  Some environments will not need
any masquerading at all.  Others, such as GCE, will not allow pod IPs to send
traffic to the internet, but have no problem with them inside your GCE Project.

### Other

- Enable auto-upgrades for your OS package manager, if desired.
- Configure log rotation for all node components (e.g. using [logrotate](http://linux.die.net/man/8/logrotate)).
- Setup liveness-monitoring (e.g. using [monit](http://linux.die.net/man/1/monit)).
- Setup volume plugin support (optional)
  - Install any client binaries for optional volume types, such as `glusterfs-client` for GlusterFS
    volumes.

### Using Configuration Management

The previous steps all involved "conventional" system administration techniques for setting up
machines.  You may want to use a Configuration Management system to automate the node configuration
process.  There are examples of [Saltstack](../admin/salt.md), Ansible, Juju, and CoreOS Cloud Config in the
various Getting Started Guides.

## Bootstrapping the Cluster

While the basic node services (kubelet, kube-proxy, docker) are typically started and managed using
traditional system administration/automation approaches, the remaining *master* components of Kubernetes are
all configured and managed *by Kubernetes*:
  - their options are specified in a Pod spec (yaml or json) rather than an /etc/init.d file or
    systemd unit.
  - they are kept running by Kubernetes rather than by init.

### etcd

You will need to run one or more instances of etcd.
  - Recommended approach: run one etcd instance, with its log written to a directory backed
    by durable storage (RAID, GCE PD)
  - Alternative: run 3 or 5 etcd instances.
    - Log can be written to non-durable storage because storage is replicated.
    - run a single apiserver which connects to one of the etc nodes.
 See [cluster-troubleshooting](../admin/cluster-troubleshooting.md) for more discussion on factors affecting cluster
availability.

To run an etcd instance:
1. copy `cluster/saltbase/salt/etcd/etcd.manifest`
1. make any modifications needed
1. start the pod by putting it into the kubelet manifest directory

### Apiserver, Controller Manager, and Scheduler

The apiserver, controller manager, and scheduler will each run as a pod on the master node.

For each of these components, the steps to start them running are similar:

1. Start with a provided template for a pod.
1. Set the `APISERVER_IMAGE`, `CNTRLMNGR_IMAGE`, and `SCHEDULER_IMAGE to the values chosen in [Selecting Images](#selecting-images).
1. Determine which flags are needed for your cluster, using the advice below each template.
1. Set the `APISERVER_FLAGS`, `CNTRLMNGR_FLAGS, and `SCHEDULER_FLAGS` to the space-separated list of flags for that component.
1. Start the pod by putting the completed template into the kubelet manifest directory.
1. Verify that the pod is started.

#### Apiserver pod template

```json
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "kube-apiserver"
  },
  "spec": {
    "hostNetwork": true,
    "containers": [
      {
        "name": "kube-apiserver",
        "image": "${APISERVER_IMAGE}",
        "command": [
          "/bin/sh",
          "-c",
          "/usr/local/bin/kube-apiserver $APISERVER_FLAGS"
        ],
        "ports": [
          {
            "name": "https",
            "hostPort": 443,
            "containerPort": 443
          },
          {
            "name": "local",
            "hostPort": 8080,
            "containerPort": 8080
          }
        ],
        "volumeMounts": [
          {
            "name": "srvkube",
            "mountPath": "/srv/kubernetes",
            "readOnly": true
          },
          {
            "name": "etcssl",
            "mountPath": "/etc/ssl",
            "readOnly": true
          }
        ],
        "livenessProbe": {
          "httpGet": {
            "path": "/healthz",
            "port": 8080
          },
          "initialDelaySeconds": 15,
          "timeoutSeconds": 15
        }
      }
    ],
    "volumes": [
      {
        "name": "srvkube",
        "hostPath": {
          "path": "/srv/kubernetes"
        }
      },
      {
        "name": "etcssl",
        "hostPath": {
          "path": "/etc/ssl"
        }
      }
    ]
  }
}
```

Here are some apiserver flags you may need to set:

- `--cloud-provider=` see [cloud providers](#cloud-providers)
- `--cloud-config=` see [cloud providers](#cloud-providers)
- `--address=${MASTER_IP}` *or* `--bind-address=127.0.0.1` and `--address=127.0.0.1` if you want to run a proxy on the master node.
- `--cluster-name=$CLUSTER_NAME`
- `--service-cluster-ip-range=$SERVICE_CLUSTER_IP_RANGE`
- `--etcd-servers=http://127.0.0.1:4001`
- `--tls-cert-file=/srv/kubernetes/server.cert`
- `--tls-private-key-file=/srv/kubernetes/server.key`
- `--admission-control=$RECOMMENDED_LIST`
  - See [admission controllers](../admin/admission-controllers.md) for recommended arguments.
- `--allow-privileged=true`, only if you trust your cluster user to run pods as root.

If you are following the firewall-only security approach, then use these arguments:

- `--token-auth-file=/dev/null`
- `--insecure-bind-address=$MASTER_IP`
- `--advertise-address=$MASTER_IP`

If you are using the HTTPS approach, then set:
- `--client-ca-file=/srv/kubernetes/ca.crt`
- `--token-auth-file=/srv/kubernetes/known_tokens.csv`
- `--basic-auth-file=/srv/kubernetes/basic_auth.csv`

This pod mounts several node file system directories using the  `hostPath` volumes.  Their purposes are:
- The `/etc/ssl` mount allows the apiserver to find the SSL root certs so it can
  authenticate external services, such as a cloud provider.
  - This is not required if you do not use a cloud provider (e.g. bare-metal).
- The `/srv/kubernetes` mount allows the apiserver to read certs and credentials stored on the
  node disk.  These could instead be stored on a persistend disk, such as a GCE PD, or baked into the image.
- Optionally, you may want to mount `/var/log` as well and redirect output there (not shown in template).
  - Do this if you prefer your logs to be accessible from the root filesystem with tools like journalctl.

*TODO* document proxy-ssh setup.

##### Cloud Providers

Apiserver supports several cloud providers.

- options for `--cloud-provider` flag are `aws`, `gce`, `mesos`, `openshift`, `ovirt`, `rackspace`, `vagrant`, or unset.
- unset used for e.g. bare metal setups.
- support for new IaaS is added by contributing code [here](../../pkg/cloudprovider/)

Some cloud providers require a config file. If so, you need to put config file into apiserver image or mount through hostPath.

- `--cloud-config=` set if cloud provider requires a config file.
- Used by `aws`, `gce`, `mesos`, `openshift`, `ovirt` and `rackspace`.
- You must put config file into apiserver image or mount through hostPath.
- Cloud config file syntax is [Gcfg](https://code.google.com/p/gcfg/).
- AWS format defined by type [AWSCloudConfig](../../pkg/cloudprovider/aws/aws.go)
- There is a similar type in the corresponding file for other cloud providers.
- GCE example: search for `gce.conf` in [this file](../../cluster/gce/configure-vm.sh)

#### Scheduler pod template

Complete this template for the scheduler pod:

```json

{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "kube-scheduler"
  },
  "spec": {
    "hostNetwork": true,
    "containers": [
      {
        "name": "kube-scheduler",
        "image": "$SCHEDULER_IMAGE",
        "command": [
          "/bin/sh",
          "-c",
          "/usr/local/bin/kube-scheduler --master=127.0.0.1:8080 $SCHEDULER_FLAGS"
        ],
        "livenessProbe": {
          "httpGet": {
            "path": "/healthz",
            "port": 10251
          },
          "initialDelaySeconds": 15,
          "timeoutSeconds": 15
        }
      }
    ]
  }
}

```

Typically, no additional flags are required for the scheduler.

Optionally, you may want to mount `/var/log` as well and redirect output there.

#### Controller Manager Template

Template for controller manager pod:

```json

{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "kube-controller-manager"
  },
  "spec": {
    "hostNetwork": true,
    "containers": [
      {
        "name": "kube-controller-manager",
        "image": "$CNTRLMNGR_IMAGE",
        "command": [
          "/bin/sh",
          "-c",
          "/usr/local/bin/kube-controller-manager $CNTRLMNGR_FLAGS"
        ],
        "volumeMounts": [
          {
            "name": "srvkube",
            "mountPath": "/srv/kubernetes",
            "readOnly": true
          },
          {
            "name": "etcssl",
            "mountPath": "/etc/ssl",
            "readOnly": true
          }
        ],
        "livenessProbe": {
          "httpGet": {
            "path": "/healthz",
            "port": 10252
          },
          "initialDelaySeconds": 15,
          "timeoutSeconds": 15
        }
      }
    ],
    "volumes": [
      {
        "name": "srvkube",
        "hostPath": {
          "path": "/srv/kubernetes"
        }
      },
      {
        "name": "etcssl",
        "hostPath": {
          "path": "/etc/ssl"
        }
      }
    ]
  }
}

```

Flags to consider using with controller manager:
 - `--cluster-name=$CLUSTER_NAME`
 - `--cluster-cidr=`
   - *TODO*: explain this flag.
 - `--allocate-node-cidrs=`
   - *TODO*: explain when you want controller to do this and when you want to do it another way.
 - `--cloud-provider=` and `--cloud-config` as described in apiserver section.
 - `--service-account-private-key-file=/srv/kubernetes/server.key`, used by the [service account](../user-guide/service-accounts.md) feature.
 - `--master=127.0.0.1:8080`

#### Starting and Verifying Apiserver, Scheduler, and Controller Manager

Place each completed pod template into the kubelet config dir
(whatever `--config=` argument of kubelet is set to, typically
`/etc/kubernetes/manifests`).  The order does not matter: scheduler and
controller manager will retry reaching the apiserver until it is up.

Use `ps` or `docker ps` to verify that each process has started.  For example, verify that kubelet has started a container for the apiserver like this:

```console
$ sudo docker ps | grep apiserver:
5783290746d5        gcr.io/google_containers/kube-apiserver:e36bf367342b5a80d7467fd7611ad873            "/bin/sh -c '/usr/lo'"    10 seconds ago      Up 9 seconds                              k8s_kube-apiserver.feb145e7_kube-apiserver-kubernetes-master_default_eaebc600cf80dae59902b44225f2fc0a_225a4695
```

Then try to connect to the apiserver:

```console
$ echo $(curl -s http://localhost:8080/healthz)
ok
$ curl -s http://localhost:8080/api
{
  "versions": [
    "v1"
  ]
}
```

If you have selected the `--register-node=true` option for kubelets, they will now begin self-registering with the apiserver.
You should soon be able to see all your nodes by running the `kubect get nodes` command.
Otherwise, you will need to manually create node objects.

### Logging

**TODO** talk about starting Logging.

### Monitoring

**TODO** talk about starting Logging.

### DNS

**TODO** talk about starting DNS.

## Troubleshooting

### Running validate-cluster

**TODO** explain how to use `cluster/validate-cluster.sh`

### Inspect pods and services

Try to run through the "Inspect your cluster" section in one of the other Getting Started Guides, such as [GCE](gce.md#inspect-your-cluster).
You should see some services.  You should also see "mirror pods" for the apiserver, scheduler and controller-manager, plus any add-ons you started.

### Try Examples

At this point you should be able to run through one of the basic examples, such as the [nginx example](../../examples/simple-nginx.md).

### Running the Conformance Test

You may want to try to run the [Conformance test](http://releases.k8s.io/HEAD/hack/conformance-test.sh).  Any failures may give a hint as to areas that need more attention.

### Networking

The nodes must be able to connect to each other using their private IP. Verify this by
pinging or SSH-ing from one node to another.

### Getting Help

If you run into trouble, please see the section on [troubleshooting](gce.md#troubleshooting), post to the
[google-containers group](https://groups.google.com/forum/#!forum/google-containers), or come ask questions on IRC at [#google-containers](http://webchat.freenode.net/?channels=google-containers) on freenode.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/scratch.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
