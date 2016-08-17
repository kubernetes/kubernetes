<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/devel/developer-guides/vagrant.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Getting started with Vagrant

Running Kubernetes with Vagrant is an easy way to run/test/develop on your
local machine in an environment using the same setup procedures when running on
GCE or AWS cloud providers. This provider is not tested on a per PR basis, if
you experience bugs when testing from HEAD, please open an issue.

### Prerequisites

1. Install latest version >= 1.8.1 of vagrant from
http://www.vagrantup.com/downloads.html

2. Install a virtual machine host. Examples:
   1. [Virtual Box](https://www.virtualbox.org/wiki/Downloads)
   2. [VMWare Fusion](https://www.vmware.com/products/fusion/) plus
[Vagrant VMWare Fusion provider](https://www.vagrantup.com/vmware)
   3. [Parallels Desktop](https://www.parallels.com/products/desktop/)
plus
[Vagrant Parallels provider](https://parallels.github.io/vagrant-parallels/)

3. Get or build a
[binary release](../../../docs/getting-started-guides/binary_release.md)

### Setup

Setting up a cluster is as simple as running:

```shell
export KUBERNETES_PROVIDER=vagrant
curl -sS https://get.k8s.io | bash
```

Alternatively, you can download
[Kubernetes release](https://github.com/kubernetes/kubernetes/releases) and
extract the archive. To start your local cluster, open a shell and run:

```shell
cd kubernetes

export KUBERNETES_PROVIDER=vagrant
./cluster/kube-up.sh
```

The `KUBERNETES_PROVIDER` environment variable tells all of the various cluster
management scripts which variant to use. If you forget to set this, the
assumption is you are running on Google Compute Engine.

By default, the Vagrant setup will create a single master VM (called
kubernetes-master) and one node (called kubernetes-node-1). Each VM will take 1
GB, so make sure you have at least 2GB to 4GB of free memory (plus appropriate
free disk space).

Vagrant will provision each machine in the cluster with all the necessary
components to run Kubernetes. The initial setup can take a few minutes to
complete on each machine.

If you installed more than one Vagrant provider, Kubernetes will usually pick
the appropriate one. However, you can override which one Kubernetes will use by
setting the
[`VAGRANT_DEFAULT_PROVIDER`](https://docs.vagrantup.com/v2/providers/default.html)
environment variable:

```shell
export VAGRANT_DEFAULT_PROVIDER=parallels
export KUBERNETES_PROVIDER=vagrant
./cluster/kube-up.sh
```

By default, each VM in the cluster is running Fedora.

To access the master or any node:

```shell
vagrant ssh master
vagrant ssh node-1
```

If you are running more than one node, you can access the others by:

```shell
vagrant ssh node-2
vagrant ssh node-3
```

Each node in the cluster installs the docker daemon and the kubelet.

The master node instantiates the Kubernetes master components as pods on the
machine.

To view the service status and/or logs on the kubernetes-master:

```shell
[vagrant@kubernetes-master ~] $ vagrant ssh master
[vagrant@kubernetes-master ~] $ sudo su

[root@kubernetes-master ~] $ systemctl status kubelet
[root@kubernetes-master ~] $ journalctl -ru kubelet

[root@kubernetes-master ~] $ systemctl status docker
[root@kubernetes-master ~] $ journalctl -ru docker

[root@kubernetes-master ~] $ tail -f /var/log/kube-apiserver.log
[root@kubernetes-master ~] $ tail -f /var/log/kube-controller-manager.log
[root@kubernetes-master ~] $ tail -f /var/log/kube-scheduler.log
```

To view the services on any of the nodes:

```shell
[vagrant@kubernetes-master ~] $ vagrant ssh node-1
[vagrant@kubernetes-master ~] $ sudo su

[root@kubernetes-master ~] $ systemctl status kubelet
[root@kubernetes-master ~] $ journalctl -ru kubelet

[root@kubernetes-master ~] $ systemctl status docker
[root@kubernetes-master ~] $ journalctl -ru docker
```

### Interacting with your Kubernetes cluster with Vagrant.

With your Kubernetes cluster up, you can manage the nodes in your cluster with
the regular Vagrant commands.

To push updates to new Kubernetes code after making source changes:

```shell
./cluster/kube-push.sh
```

To stop and then restart the cluster:

```shell
vagrant halt
./cluster/kube-up.sh
```

To destroy the cluster:

```shell
vagrant destroy
```

Once your Vagrant machines are up and provisioned, the first thing to do is to
check that you can use the `kubectl.sh` script.

You may need to build the binaries first, you can do this with `make`

```shell
$ ./cluster/kubectl.sh get nodes
```

### Authenticating with your master

When using the vagrant provider in Kubernetes, the `cluster/kubectl.sh` script
will cache your credentials in a `~/.kubernetes_vagrant_auth` file so you will
not be prompted for them in the future.

```shell
cat ~/.kubernetes_vagrant_auth
```

```json
{ "User": "vagrant",
  "Password": "vagrant",
  "CAFile": "/home/k8s_user/.kubernetes.vagrant.ca.crt",
  "CertFile": "/home/k8s_user/.kubecfg.vagrant.crt",
  "KeyFile": "/home/k8s_user/.kubecfg.vagrant.key"
}
```

You should now be set to use the `cluster/kubectl.sh` script. For example try to
list the nodes that you have started with:

```shell
./cluster/kubectl.sh get nodes
```

### Running containers

You can use `cluster/kube-*.sh` commands to interact with your VM machines:

```shell
$ ./cluster/kubectl.sh get pods
NAME        READY     STATUS    RESTARTS   AGE

$ ./cluster/kubectl.sh get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE

$ ./cluster/kubectl.sh get deployments
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR   REPLICAS
```

To Start a container running nginx with a Deployment and three replicas:

```shell
$ ./cluster/kubectl.sh run my-nginx --image=nginx --replicas=3 --port=80
```

When listing the pods, you will see that three containers have been started and
are in Waiting state:

```shell
$ ./cluster/kubectl.sh get pods
NAME                        READY     STATUS              RESTARTS   AGE
my-nginx-3800858182-4e6pe   0/1       ContainerCreating   0          3s
my-nginx-3800858182-8ko0s   1/1       Running             0          3s
my-nginx-3800858182-seu3u   0/1       ContainerCreating   0          3s
```

When the provisioning is complete:

```shell
$ ./cluster/kubectl.sh get pods
NAME                        READY     STATUS    RESTARTS   AGE
my-nginx-3800858182-4e6pe   1/1       Running   0          40s
my-nginx-3800858182-8ko0s   1/1       Running   0          40s
my-nginx-3800858182-seu3u   1/1       Running   0          40s

$ ./cluster/kubectl.sh get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE

$ ./cluster/kubectl.sh get deployments
NAME       DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
my-nginx   3         3         3            3           1m
```

We did not start any Services, hence there are none listed. But we see three
replicas displayed properly. Check the
[guestbook](https://github.com/kubernetes/kubernetes/tree/%7B%7Bpage.githubbranch%7D%7D/examples/guestbook)
application to learn how to create a Service. You can already play with scaling
the replicas with:

```shell
$ ./cluster/kubectl.sh scale deployments my-nginx --replicas=2
$ ./cluster/kubectl.sh get pods
NAME                        READY     STATUS    RESTARTS   AGE
my-nginx-3800858182-4e6pe   1/1       Running   0          2m
my-nginx-3800858182-8ko0s   1/1       Running   0          2m
```

Congratulations!

### Testing

The following will run all of the end-to-end testing scenarios assuming you set
your environment:

```shell
NUM_NODES=3 go run hack/e2e.go -v --build --up --test --down
```

### Troubleshooting

#### I keep downloading the same (large) box all the time!

By default the Vagrantfile will download the box from S3. You can change this
(and cache the box locally) by providing a name and an alternate URL when
calling `kube-up.sh`

```shell
export KUBERNETES_BOX_NAME=choose_your_own_name_for_your_kuber_box
export KUBERNETES_BOX_URL=path_of_your_kuber_box
export KUBERNETES_PROVIDER=vagrant
./cluster/kube-up.sh
```

#### I am getting timeouts when trying to curl the master from my host!

During provision of the cluster, you may see the following message:

```shell
Validating node-1
.............
Waiting for each node to be registered with cloud provider
error: couldn't read version from server: Get https://10.245.1.2/api: dial tcp 10.245.1.2:443: i/o timeout
```

Some users have reported VPNs may prevent traffic from being routed to the host
machine into the virtual machine network.

To debug, first verify that the master is binding to the proper IP address:

```
$ vagrant ssh master
$ ifconfig | grep eth1 -C 2
eth1: flags=4163<UP,BROADCAST,RUNNING,MULTICAST> mtu 1500 inet 10.245.1.2 netmask
   255.255.255.0 broadcast 10.245.1.255
```

Then verify that your host machine has a network connection to a bridge that can
serve that address:

```shell
$ ifconfig | grep 10.245.1 -C 2

vboxnet5: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.245.1.1  netmask 255.255.255.0  broadcast 10.245.1.255
        inet6 fe80::800:27ff:fe00:5  prefixlen 64  scopeid 0x20<link>
        ether 0a:00:27:00:00:05  txqueuelen 1000  (Ethernet)
```

If you do not see a response on your host machine, you will most likely need to
connect your host to the virtual network created by the virtualization provider.

If you do see a network, but are still unable to ping the machine, check if your
VPN is blocking the request.

#### I just created the cluster, but I am getting authorization errors!

You probably have an incorrect ~/.kubernetes_vagrant_auth file for the cluster
you are attempting to contact.

```shell
rm ~/.kubernetes_vagrant_auth
```

After using kubectl.sh make sure that the correct credentials are set:

```shell
cat ~/.kubernetes_vagrant_auth
```

```json
{
  "User": "vagrant",
  "Password": "vagrant"
}
```

#### I just created the cluster, but I do not see my container running!

If this is your first time creating the cluster, the kubelet on each node
schedules a number of docker pull requests to fetch prerequisite images. This
can take some time and as a result may delay your initial pod getting
provisioned.

#### I have Vagrant up but the nodes won't validate!

Log on to one of the nodes (`vagrant ssh node-1`) and inspect the salt minion
log (`sudo cat /var/log/salt/minion`).

#### I want to change the number of nodes!

You can control the number of nodes that are instantiated via the environment
variable `NUM_NODES` on your host machine. If you plan to work with replicas, we
strongly encourage you to work with enough nodes to satisfy your largest
intended replica size. If you do not plan to work with replicas, you can save
some system resources by running with a single node. You do this, by setting
`NUM_NODES` to 1 like so:

```shell
export NUM_NODES=1
```

#### I want my VMs to have more memory!

You can control the memory allotted to virtual machines with the
`KUBERNETES_MEMORY` environment variable. Just set it to the number of megabytes
you would like the machines to have. For example:

```shell
export KUBERNETES_MEMORY=2048
```

If you need more granular control, you can set the amount of memory for the
master and nodes independently. For example:

```shell
export KUBERNETES_MASTER_MEMORY=1536
export KUBERNETES_NODE_MEMORY=2048
```

#### I want to set proxy settings for my Kubernetes cluster boot strapping!

If you are behind a proxy, you need to install the Vagrant proxy plugin and set
the proxy settings:

```shell
vagrant plugin install vagrant-proxyconf
export KUBERNETES_HTTP_PROXY=http://username:password@proxyaddr:proxyport
export KUBERNETES_HTTPS_PROXY=https://username:password@proxyaddr:proxyport
```

You can also specify addresses that bypass the proxy, for example:

```shell
export KUBERNETES_NO_PROXY=127.0.0.1
```

If you are using sudo to make Kubernetes build, use the `-E` flag to pass in the
environment variables. For example, if running `make quick-release`, use:

```shell
sudo -E make quick-release
```

#### I have repository access errors during VM provisioning!

Sometimes VM provisioning may fail with errors that look like this:

```
Timeout was reached for https://mirrors.fedoraproject.org/metalink?repo=fedora-23&arch=x86_64 [Connection timed out after 120002 milliseconds]
```

You may use a custom Fedora repository URL to fix this:

```shell
export CUSTOM_FEDORA_REPOSITORY_URL=https://download.fedoraproject.org/pub/fedora/
```

#### I ran vagrant suspend and nothing works!

`vagrant suspend` seems to mess up the network. It's not supported at this time.

#### I want vagrant to sync folders via nfs!

You can ensure that vagrant uses nfs to sync folders with virtual machines by
setting the KUBERNETES_VAGRANT_USE_NFS environment variable to 'true'. nfs is
faster than virtualbox or vmware's 'shared folders' and does not require guest
additions. See the
[vagrant docs](http://docs.vagrantup.com/v2/synced-folders/nfs.html) for details
on configuring nfs on the host. This setting will have no effect on the libvirt
provider, which uses nfs by default. For example:

```shell
export KUBERNETES_VAGRANT_USE_NFS=true
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/developer-guides/vagrant.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
