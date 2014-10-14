# CoreOS Cloud Configs

The recommended way to run Kubernetes on CoreOS is to use [Cloud-Config](https://coreos.com/docs/cluster-management/setup/cloudinit-cloud-config/).

## Setup

Get the cloud-config templates which we'll be editing in place for this example.
```
git clone https://github.com/GoogleCloudPlatform/kubernetes.git
cd kubernetes/docs/getting-started-guides/coreos/configs
```

### Standalone

The standalone cloud-config file can be used to setup a single node Kubernetes cluster that has had CoreOS installed.

* [standalone.yml](configs/standalone.yml)

Skip to ['Configure Access'](#configure-access).


### Cluster 

These are the current instructions for [Kelsey Hightowers blog post Running Kubernetes on CoreOS Part 2](https://coreos.com/blog/running-kubernetes-example-on-CoreOS-part-2/)
which provides a good background context for understanding Kubernetes and how to set this up using VMWare Fusion Pro.
 
#### Machine Configuration
To start we'll need 3 nodes for our cluster with the following:
*  1 CPU
*  512 MB RAM
*  20 GB HDD
*  2 Network Interfaces
*  CD ROM (to install CoreOS and to provide configuration from [cloud-drive] (http://coreos.com/docs/cluster-management/setup/cloudinit-config-drive/)
 
The primary network interface for each machine should be on a network with access to the outside world in order to 
update CoreOS, access the Docker repository, download Kubernetes, etc. The second interface on each machine should each 
be connected to a switch. (VMWare Fusion Pro users can create a custom network with DHCP and NAT disabled that these 
secondary interfaces connected to - see [the blog post](https://coreos.com/blog/running-kubernetes-example-on-CoreOS-part-2/)
 for screenshots).

Boot each node from the [CoreOS](https://coreos.com/) ISO. Hit 'Return' a few times in the console window of a node. 
Above the login prompt CoreOS lists the names it has generated for the network interfaces followed by the IP address it 
has been assigned. Note the first interface name, and proceed to install CoreOS. 

#### Cloud Configuration
The following cloud-config templates are used to setup a three node Kubernetes cluster.
* [master.yml](configs/master.yml)
* [node1.yml](configs/node1.yml)
* [node2.yml](configs/node2.yml)

Search for occurrences of 'ens33' in these templates and replace with the interface name provided by CoreOS.

Replace all occurrences of '192.168.12.10' with the IP address you wish to apply to the master node, '192.168.12.11' 
with the IP address to assign to node1, '192.168.12.12' with IP address for node2. In the section 
'coreos/units/static.network' set the DNS and Gateway entries to match your network. (VMWare Fusion users can find this 
information in: "/Library/Preferences/VMware Fusion/vmnet8/dhcpd.conf" see [the blog post](https://coreos.com/blog/running-kubernetes-example-on-CoreOS-part-2/)
 - be sure to choose static IPs outside the dynamic ip range specified here).

### Configure Access

For both the standalone and cluster configurations, the final change required to the cloud-config file(s) is to replace 
<ssh_public_key> with your public ssh key (typically the contents of ~/.ssh/id_rsa.pub).

### Create config-drives

Now create the ISO images that cloud-config will access when booting your node(s). 

```
mkdir -p /tmp/new-drive/openstack/latest/
mkdir -p ~/iso
```

Using Linux:

```
for i in standalone master node1 node2; do
  cp ${i}.yml /tmp/new-drive/openstack/latest/user_data
  mkisofs -R -V config-2 -o ~/iso/${i}.iso /tmp/new-drive
done
```

Using OS X:

```
for i in standalone master node1 node2; do
  cp ${i}.yml /tmp/new-drive/openstack/latest/user_data
  hdiutil makehybrid -iso -joliet -joliet-volume-name "config-2" -joliet -o ~/iso/${i}.iso /tmp/new-drive
done
```

Make each ISO file accessible to its corresponding node by using it to define a cd/dvd drive for the VM (or create a 
physical CD for bare metal), and boot the node. At the consoles login prompt, confirm the configured IP address for the 
node is listed next to the interface name.

## Remote Access

Setup a SSH tunnel to the Kubernetes API Server, replacing ${APISERVER} with the IP address of your master or 
standalone node. 

```
sudo ssh -f -nNT -L 8080:127.0.0.1:8080 core@${APISERVER}
```

Download a kubecfg client

**Darwin**

```
wget http://storage.googleapis.com/kubernetes/darwin/kubecfg -O /usr/local/bin/kubecfg
```

**Linux**

```
wget http://storage.googleapis.com/kubernetes/kubecfg -O /usr/local/bin/kubecfg
```

Issue commands remotely using the kubecfg command line tool.

```
kubecfg list /pods
```

Test a sample pod:

````
kubecfg -c examples/guestbook-go/redis-master-pod.json create pods
kubecfg list /pods
```

Your pod should now be listed as 'running'.
