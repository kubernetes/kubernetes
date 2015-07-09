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

- [Designing and Preparing](http://releases.k8s.io/HEAD/docs/getting-started-guides/#designing-and-preparing)
    - [Learning](http://releases.k8s.io/HEAD/docs/getting-started-guides/#learning)
    - [Cloud Provider](http://releases.k8s.io/HEAD/docs/getting-started-guides/#cloud-provider)
    - [Nodes](http://releases.k8s.io/HEAD/docs/getting-started-guides/#nodes)
    - [Network](http://releases.k8s.io/HEAD/docs/getting-started-guides/#network)
    - [Cluster Naming](http://releases.k8s.io/HEAD/docs/getting-started-guides/#cluster-naming)
    - [Software Binaries](http://releases.k8s.io/HEAD/docs/getting-started-guides/#software-binaries)
        - [Downloading and Extracting Kubernetes Binaries](http://releases.k8s.io/HEAD/docs/getting-started-guides/#downloading-and-extracting-kubernetes-binaries)
        - [Selecting Images](http://releases.k8s.io/HEAD/docs/getting-started-guides/#selecting-images)
    - [Security Models](http://releases.k8s.io/HEAD/docs/getting-started-guides/#security-models)
        - [Preparing Certs](http://releases.k8s.io/HEAD/docs/getting-started-guides/#preparing-certs)
        - [Preparing Credentials](http://releases.k8s.io/HEAD/docs/getting-started-guides/#preparing-credentials)
- [Configuring and Installing Base Software on Nodes](http://releases.k8s.io/HEAD/docs/getting-started-guides/#configuring-and-installing-base-software-on-nodes)
    - [Docker](http://releases.k8s.io/HEAD/docs/getting-started-guides/#docker)
    - [rkt](http://releases.k8s.io/HEAD/docs/getting-started-guides/#rkt)
    - [kubelet](http://releases.k8s.io/HEAD/docs/getting-started-guides/#kubelet)
    - [kube-proxy](http://releases.k8s.io/HEAD/docs/getting-started-guides/#kube-proxy)
    - [Networking](http://releases.k8s.io/HEAD/docs/getting-started-guides/#networking)
    - [Other](http://releases.k8s.io/HEAD/docs/getting-started-guides/#other)
    - [Using Configuration Management](http://releases.k8s.io/HEAD/docs/getting-started-guides/#using-configuration-management)
- [Bootstrapping the Cluster](http://releases.k8s.io/HEAD/docs/getting-started-guides/#bootstrapping-the-cluster)
    - [etcd](http://releases.k8s.io/HEAD/docs/getting-started-guides/#etcd)
    - [Apiserver](http://releases.k8s.io/HEAD/docs/getting-started-guides/#apiserver)
        - [Apiserver pod template](http://releases.k8s.io/HEAD/docs/getting-started-guides/#apiserver-pod-template)
    - [Starting Apiserver](http://releases.k8s.io/HEAD/docs/getting-started-guides/#starting-apiserver)
    - [Scheduler](http://releases.k8s.io/HEAD/docs/getting-started-guides/#scheduler)
    - [Controller Manager](http://releases.k8s.io/HEAD/docs/getting-started-guides/#controller-manager)
    - [DNS](http://releases.k8s.io/HEAD/docs/getting-started-guides/#dns)
    - [Logging](http://releases.k8s.io/HEAD/docs/getting-started-guides/#logging)
    - [Monitoring](http://releases.k8s.io/HEAD/docs/getting-started-guides/#monitoring)
    - [Miscellaneous Resources](http://releases.k8s.io/HEAD/docs/getting-started-guides/#miscelaneous-resources)
- [Troubleshooting](http://releases.k8s.io/HEAD/docs/getting-started-guides/#troubleshooting)
    - [Running validate-cluster](http://releases.k8s.io/HEAD/docs/getting-started-guides/#running-validate-cluster)
    - [Inspect pods and services](http://releases.k8s.io/HEAD/docs/getting-started-guides/#inspect-pods-and-services)
    - [Try Examples](http://releases.k8s.io/HEAD/docs/getting-started-guides/#try-examples)
    - [Running the Conformance Test](http://releases.k8s.io/HEAD/docs/getting-started-guides/#running-the-conformance-test)
    - [Networking](http://releases.k8s.io/HEAD/docs/getting-started-guides/#networking)
    - [Getting Help](http://releases.k8s.io/HEAD/docs/getting-started-guides/#getting-help)

## Designing and Preparing

### Learning 
  1. You should be familiar with using Kubernetes already.  We suggest you set
    up a temporary cluster by following one of the other Getting Started Guides.
    This will help you become familiar with the CLI ([kubectl](http://releases.k8s.io/HEAD/docs/getting-started-guides/../kubectl.md)) and concepts ([pods](http://releases.k8s.io/HEAD/docs/getting-started-guides/../pods.md), [services](http://releases.k8s.io/HEAD/docs/getting-started-guides/../services.md), etc.) first.
  1. You should have `kubectl` installed on your desktop.  This will happen as a side
    effect of completing one of the other Getting Started Guides.

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
Kubernetes has a distinctive [networking model](http://releases.k8s.io/HEAD/docs/getting-started-guides/../networking.md).

Kubernetes allocates an IP address to each pod.  When creating a cluster, you
need to allocate a block of IPs for Kubernetes to use as Pod IPs.  The simplest
approach is to allocate a different block of IPs to each node in the cluster as
the node is added.  A process in one pod should be able to communicate with
another pod using the IP of the second pod.  This connectivity can be
accomplished in two ways:
- Configure network to route Pod IPs
  - Harder to setup from scratch.
  - Google Compute Engine ([GCE](http://releases.k8s.io/HEAD/docs/getting-started-guides/gce.md)) and [AWS](http://releases.k8s.io/HEAD/docs/getting-started-guides/aws.md) guides use this approach.
  - Need to make the Pod IPs routable by programming routers, switches, etc.
  - Can be configured external to kubernetes, or can implement in the "Routes" interface of a Cloud Provider module.
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

Kubernetes also allocates an IP to each [service](http://releases.k8s.io/HEAD/docs/getting-started-guides/../services.md).  However,
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
[Developer Documentation](http://releases.k8s.io/HEAD/docs/getting-started-guides/ ../devel/README.md).  Only using a binary release is covered in this guide.

Download the [latest binary release](
https://github.com/GoogleCloudPlatform/kubernetes/releases/latest) and unzip it.
Then locate `./kubernetes/server/kubernetes-server-linux-amd64.tar.gz` and unzip *that*.
Then, within the second set of unzipped files, locate `./kubernetes/server/bin`, which contains
all the necessary binaries.

#### Selecting Images
You will run docker, kubelet, and kube-proxy outside of a container, the same way you would run any system daemon, so
you just need the bare binaries.  For etcd, kube-apiserver, kube-controller-manager, and kube-scheduler, 
we recommend that you run these as containers, so you need an image to be built.

You have several choices for Kubernetes images:
1. Use images hosted on Google Container Registry (GCR):
  - e.g `gcr.io/google_containers/kube-apiserver:$TAG`, where `TAG` is the latest
    release tag, which can be found on the [latest releases page](
    https://github.com/GoogleCloudPlatform/kubernetes/releases/latest). 
  - Ensure $TAG is the same tag as the release tag you are using for kubelet and kube-proxy.
- Build your own images.
  - Useful if you are using a private registry.
  - The release contains files such as `./kubernetes/server/bin/kube-apiserver.tar` which
    can be converted into docker images using a command like
    `tar -C kube-apiserver -c . | docker import - kube-apiserver`
  - *TODO*: test above command.

For etcd, you can:
- Use images hosted on Google Container Registry (GCR), such as `gcr.io/google_containers/etcd:2.0.12`
- Use images hosted on [Docker Hub](https://registry.hub.docker.com/u/coreos/etcd/) or [quay.io](https://registry.hub.docker.com/u/coreos/etcd/)
- Use etcd binary included in your OS distro.
- Build your own image
  - You can do: `cd kubernetes/cluster/images/etcd; make`

We recommend that you use the etcd version which is provided in the kubernetes binary distribution.   The kubernetes binaries in the release
were tested extensively with this version of etcd and not with any other version.
The recommended version number can also be found as the value of `ETCD_VERSION` in `kubernetes/cluster/images/etcd/Makefile`.

The remainder of the document assumes that the image identifiers have been chosen and stored in corresponding env vars.  Examples (replace with latest tags and appropriate registry):
  - `APISERVER_IMAGE=gcr.io/google_containers/kube-apiserver:$TAG`
  - `SCHEDULER_IMAGE=gcr.io/google_containers/kube-scheduler:$TAG`
  - `CNTRLMNGR_IMAGE=gcr.io/google_containers/kube-controller-manager:$TAG`
  - `ETCD_IMAGE=gcr.io/google_containers/etcd:$ETCD_VERSION`

### Security Models

There are two main options for security:
1. Access the apiserver using HTTP.
  - Use a firewall for security.
  - This is easier to setup.
1. Access the apiserver using HTTPS  
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
The format for this file is described in the [authentication documentation](
../authentication.md).

For distributing credentials to clients, the convention in Kubernetes is to put the credentials
into a [kubeconfig file](http://releases.k8s.io/HEAD/docs/getting-started-guides/../kubeconfig-file.md).

The kubeconfig file for the administrator can be created as follows:
 - If you have already used Kubernetes with a non-custom cluster (for example, used a Getting Started
   Guide), you will already have a `$HOME/.kube/config` file.
 - You need to add certs, keys, and the master IP to the kubeconfig file:
    - If using the firewall-only security option, set the apiserver this way:
      - `kubectl config set-cluster $CLUSTER_NAME --server=http://$MASTER_IP --insecure-skip-tls-verify=true`
    - Otherwise, do this to set the apiserver ip, client certs, and user credentials.
      - `kubectl config set-cluster $CLUSTER_NAME --certificate-authority=$CA_CERT --embed-certs=true --server=https://$MASTER_IP`
      - `kubectl config set-credentials $CLUSTER_NAME --client-certificate=$CLI_CERT --client-key=$CLI_KEY --embed-certs=true --token=$TOKEN`
    - Set your cluster as the default cluster to use:
      - `kubectl config set-context $CLUSTER_NAME --cluster=$CLUSTER_NAME --user=admin`
      - `kubectl config use-context $CONTEXT  --cluster=$CONTEXT`

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
```
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

This section discusses how to configure machines to be kubernetes nodes. 

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
```
iptables -t nat -F 
ifconfig docker0 down
brctl delbr docker0
```

The way you configure docker will depend in whether you have chosen the routable-vip or overlay-network approaches for your network.
Some docker options will want to think about:
  - create your own bridge for the per-node CIDR ranges, and set `--bridge=cbr0` and `--bip=false`.  Or let docker do it with `--bip=true`.
  - `--iptables=false` so docker will not manipulate iptables for host-ports (too coarse on older docker versions, may be fixed in newer versions)
so that kube-proxy can manage iptables instead of docker.
  - `--ip-masq=false`
    - if you have setup PodIPs to be routable, then you want this false, otherwise, docker will
      rewrite the PodIP source-address to a NodeIP.
    - some environments (e.g. GCE) still need you to masquerade out-bound traffic when it leaves the cloud environment. This is very environment specific.
    - if you are using an overlay network, consult those instructions.  
  - `--bip=`
    - should be the CIDR range for pods for that specific node.
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

[rkt](https://github.com/coreos/rkt) is an alterative to Docker.  You only need to install one of Docker or rkt.

*TODO*: how to install and configure rkt.

### kubelet

All nodes should run kubelet.  See [Selecting Binaries](http://releases.k8s.io/HEAD/docs/getting-started-guides/#selecting-binaries).

Arguments to consider:
  - If following the HTTPS security approach:
    - `--api-servers=https://$MASTER_IP`
    - `--kubeconfig=/var/lib/kubelet/kubeconfig`
  - Otherwise, if taking the firewall-based security approach
    - `--api-servers=http://$MASTER_IP`
  - `--config=/etc/kubernetes/manifests` -%}
  - `--cluster-dns=` to the address of the DNS server you will setup (see [Starting Addons](http://releases.k8s.io/HEAD/docs/getting-started-guides/#starting-addons).)
  - `--cluster-domain=` to the dns domain prefix to use for cluster DNS addresses.
  - `--docker-root=`
  - `--root-dir=`
  - `--configure-cbr0=` (described above)
  - `--register-node` (described in [Node](http://releases.k8s.io/HEAD/docs/getting-started-guides/../node.md) documentation.

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
further in the [networking documentation](http://releases.k8s.io/HEAD/docs/getting-started-guides/../networking.md).  The bridge itself
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
```iptables -w -t nat -A POSTROUTING -o eth0 -j MASQUERADE \! -d ${CLUSTER_SUBNET}```
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
process.  There are examples of [Saltstack](http://releases.k8s.io/HEAD/docs/getting-started-guides/../salt.md), Ansible, Juju, and CoreOS Cloud Config in the
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
 See [Availability](http://releases.k8s.io/HEAD/docs/getting-started-guides/../availability.md) for more discussion on factors affecting cluster
availability.

To run an etcd instance:
1. copy `cluster/saltbase/salt/etcd/etcd.manifest`
1. make any modifications needed
1. start the pod by putting it into the kubelet manifest directory

### Apiserver

To run the apiserver:
1. select the correct flags for your cluster
1. write a pod spec for the apiserver using the provided template
1. start the pod by putting it into the kubelet manifest directory

Here are some apiserver flags you may need to set:
  - `--cloud-provider=`
  - `--cloud-config=` if cloud provider requires a config file (GCE, AWS). If so, need to put config file into apiserver image or mount through hostPath.
  - `--address=${MASTER_IP}`.  
   - or `--bind-address=127.0.0.1` and `--address=127.0.0.1` if you want to run a proxy on the master node.
  - `--cluster-name=$CLUSTER_NAME`
  - `--service-cluster-ip-range=$SERVICE_CLUSTER_IP_RANGE`
  - `--etcd-servers=http://127.0.0.1:4001`
  - `--tls-cert-file=/srv/kubernetes/server.cert` -%}
  - `--tls-private-key-file=/srv/kubernetes/server.key` -%}
  - `--admission-control=$RECOMMENDED_LIST`
    - See [admission controllers](http://releases.k8s.io/HEAD/docs/getting-started-guides/../admission_controllers.md) for recommended arguments.
  - `--allow-privileged=true`, only if you trust your cluster user to run pods as root.
 
If you are following the firewall-only security approach, then use these arguments:
  - `--token-auth-file=/dev/null`
  - `--insecure-bind-address=$MASTER_IP`
  - `--advertise-address=$MASTER_IP`

If you are using the HTTPS approach, then set:
  - `--client-ca-file=/srv/kubernetes/ca.crt`
  - `--token-auth-file=/srv/kubernetes/known_tokens.csv`
  - `--basic-auth-file=/srv/kubernetes/basic_auth.csv`

*TODO* document proxy-ssh setup.

#### Apiserver pod template
*TODO*: convert to version v1.

```json
{
"apiVersion": "v1beta3",
"kind": "Pod",
"metadata": {"name":"kube-apiserver"},
"spec":{
"hostNetwork": true,
"containers":[
    {
    "name": "kube-apiserver",
    "image": "${APISERVER_IMAGE}",
    "command": [
                 "/bin/sh",
                 "-c",
                 "/usr/local/bin/kube-apiserver $ARGS"
               ],
    "livenessProbe": {
      "httpGet": {
        "path": "/healthz",
        "port": 8080
      },
      "initialDelaySeconds": 15,
      "timeoutSeconds": 15
    },
    "ports":[
      { "name": "https",
        "containerPort": 443,
        "hostPort": 443},
      { "name": "local",
        "containerPort": 8080,
        "hostPort": 8080}
        ],
    "volumeMounts": [
        { "name": "srvkube",
        "mountPath": "/srv/kubernetes",
        "readOnly": true},
        { "name": "etcssl",
        "mountPath": "/etc/ssl",
        "readOnly": true},
      ]
    }
],
"volumes":[
  { "name": "srvkube",
    "hostPath": {
        "path": "/srv/kubernetes"}
  },
  { "name": "etcssl",
    "hostPath": {
        "path": "/etc/ssl"}
  },
]
}}
```

The `/etc/ssl` mount allows the apiserver to find the SSL root certs so it can
authenticate external services, such as a cloud provider.

The `/srv/kubernetes` mount allows the apiserver to read certs and credentials stored on the
node disk.

Optionally, you may want to mount `/var/log` as well and redirect output there.

#### Starting Apiserver
Place the completed pod template into the kubelet config dir
(whatever `--config=` argument of kubelet is set to, typically
`/etc/kubernetes/manifests`).

Next, verify that kubelet has started a container for the apiserver:
```
$ sudo docker ps | grep apiserver:
5783290746d5        gcr.io/google_containers/kube-apiserver:e36bf367342b5a80d7467fd7611ad873            "/bin/sh -c '/usr/lo'"    10 seconds ago      Up 9 seconds                              k8s_kube-apiserver.feb145e7_kube-apiserver-kubernetes-master_default_eaebc600cf80dae59902b44225f2fc0a_225a4695   ```
```

Then try to connect to the apiserver:
```
$ echo $(curl -s http://localhost:8080/healthz)
ok
$ curl -s http://localhost:8080/api
{
  "versions": [
    "v1beta3",
    "v1"
  ]
}
```

If you have selected the `--register-node=true` option for kubelets, they will now being self-registering with the apiserver.
You should soon be able to see all your nodes by running the `kubect get nodes` command.
Otherwise, you will need to manually create node objects.

### Scheduler

*TODO*: convert to version v1.

Complete this template for the scheduler pod:
```
{
"apiVersion": "v1beta3",
"kind": "Pod",
"metadata": {"name":"kube-scheduler"},
"spec":{
"hostNetwork": true,
"containers":[
    {
    "name": "kube-scheduler",
    "image": "$SCHEDULER_IMAGE",
    "command": [
                 "/bin/sh",
                 "-c",
                 "/usr/local/bin/kube-scheduler --master=127.0.0.1:8080"
               ],
    "livenessProbe": {
      "httpGet": {
        "path": "/healthz",
        "port": 10251
      },
      "initialDelaySeconds": 15,
      "timeoutSeconds": 15
    },
    }
],
}}
```
Optionally, you may want to mount `/var/log` as well and redirect output there.

Start as described for apiserver.

### Controller Manager
To run the controller manager:
  - select the correct flags for your cluster
  - write a pod spec for the controller manager using the provided template
  - start the controller manager pod

Flags to consider using with controller manager.
 - `--cluster-name=$CLUSTER_NAME`
 - `--cluster-cidr=`
   - *TODO*: explain this flag.
 - `--allocate-node-cidrs=`
   - *TODO*: explain when you want controller to do this and when you wanna do it another way.
 - `--cloud-provider=` and `--cloud-config` as described in apiserver section.
 - `--service-account-private-key-file=/srv/kubernetes/server.key`, used by [service account](http://releases.k8s.io/HEAD/docs/getting-started-guides/../service_accounts.md) feature.  
 - `--master=127.0.0.1:8080`

Template for controller manager pod:
```
{
"apiVersion": "v1beta3",
"kind": "Pod",
"metadata": {"name":"kube-controller-manager"},
"spec":{
"hostNetwork": true,
"containers":[
    {
    "name": "kube-controller-manager",
    "image": "$CNTRLMNGR_IMAGE",
    "command": [
                 "/bin/sh",
                 "-c",
                 "/usr/local/bin/kube-controller-manager $ARGS"
               ],
    "livenessProbe": {
      "httpGet": {
        "path": "/healthz",
        "port": 10252
      },
      "initialDelaySeconds": 15,
      "timeoutSeconds": 15
    },
    "volumeMounts": [
        { "name": "srvkube",
        "mountPath": "/srv/kubernetes",
        "readOnly": true},
        { "name": "etcssl",
        "mountPath": "/etc/ssl",
        "readOnly": true},
      ]
    }
],
"volumes":[
  { "name": "srvkube",
    "hostPath": {
        "path": "/srv/kubernetes"}
  },
  { "name": "etcssl",
    "hostPath": {
        "path": "/etc/ssl"}
  },
]
}}
```


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

Try to run through the "Inspect your cluster" section in one of the other Getting Started Guides, such as [GCE](http://releases.k8s.io/HEAD/docs/getting-started-guides/gce.md#inspect-your-cluster).
You should see some services.  You should also see "mirror pods" for the apiserver, scheduler and controller-manager, plus any add-ons you started.

### Try Examples

At this point you should be able to run through one of the basic examples, such as the [nginx example](http://releases.k8s.io/HEAD/docs/getting-started-guides/../../examples/simple-nginx.md).

### Running the Conformance Test

You may want to try to run the [Conformance test](http://releases.k8s.io/HEAD/docs/getting-started-guides/../hack/conformance.sh).  Any failures may give a hint as to areas that need more attention.

### Networking

The nodes must be able to connect to each other using their private IP. Verify this by
pinging or SSH-ing from one node to another.

### Getting Help
If you run into trouble, please see the section on [troubleshooting](http://releases.k8s.io/HEAD/docs/getting-started-guides/gce.md#troubleshooting), post to the
[google-containers group](https://groups.google.com/forum/#!forum/google-containers), or come ask questions on IRC at #google-containers on freenode.



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/scratch.md?pixel)]()
