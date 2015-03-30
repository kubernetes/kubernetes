##Getting started on [Fedora](http://fedoraproject.org)

This is a getting started guide for Fedora.  It is a manual configuration so you understand all the underlying packages / services / ports, etc...

This guide will only get ONE node (previously minion) working.  Multiple nodes require a functional [networking configuration](https://github.com/GoogleCloudPlatform/lmktfy/blob/master/docs/networking.md) done outside of lmktfy.  Although the additional lmktfy configuration requirements should be obvious.

The lmktfy package provides a few services: lmktfy-apiserver, lmktfy-scheduler, lmktfy-controller-manager, lmktfylet, lmktfy-proxy.  These services are managed by systemd and the configuration resides in a central location: /etc/lmktfy.  We will break the services up between the hosts.  The first host, fed-master, will be the lmktfy master.  This host will run the lmktfy-apiserver, lmktfy-controller-manager, and lmktfy-scheduler.  In addition, the master will also run _etcd_ (not needed if _etcd_ runs on a different host but this guide assumes that _etcd_ and lmktfy master run on the same host).  The remaining host, fed-node will be the node and run lmktfylet, proxy and docker.

**System Information:**

Hosts:
```
fed-master = 192.168.121.9
fed-node = 192.168.121.65
```

**Prepare the hosts:**
    
* Install lmktfy on all hosts - fed-{master,node}.  This will also pull in etcd and docker.  This guide has been tested with lmktfy-0.12.0 but should work with later versions too.
* The [--enablerepo=update-testing](https://fedoraproject.org/wiki/QA:Updates_Testing) directive in the yum command below will ensure that the most recent LMKTFY version that is scheduled for pre-release will be installed. This should be a more recent version than the Fedora "stable" release for LMKTFY that you would get without adding the directive. 
* If you want the very latest LMKTFY release [you can download and yum install the RPM directly from Fedora Koji](http://koji.fedoraproject.org/koji/packageinfo?packageID=19202) instead of using the yum install command below.

```
yum -y install --enablerepo=updates-testing lmktfy
```

* Add master and node to /etc/hosts on all machines (not needed if hostnames already in DNS). Make sure that communication works between fed-master and fed-node by using a utility such as ping.

```
echo "192.168.121.9	fed-master
192.168.121.65	fed-node" >> /etc/hosts
```

* Edit /etc/lmktfy/config which will be the same on all hosts (master and node) to contain:

```
# Comma separated list of nodes in the etcd cluster
LMKTFY_MASTER="--master=http://fed-master:8080"

# logging to stderr means we get it in the systemd journal
LMKTFY_LOGTOSTDERR="--logtostderr=true"

# journal message level, 0 is debug
LMKTFY_LOG_LEVEL="--v=0"

# Should this cluster be allowed to run privileged docker containers
LMKTFY_ALLOW_PRIV="--allow_privileged=false"
```

* Disable the firewall on both the master and node, as docker does not play well with other firewall rule managers.  Please note that iptables-services does not exist on default fedora server install.

```
systemctl disable iptables-services firewalld
systemctl stop iptables-services firewalld
```

**Configure the lmktfy services on the master.**

* Edit /etc/lmktfy/apiserver to appear as such.  The portal_net IP addresses must be an unused block of addresses, not used anywhere else.  They do not need to be routed or assigned to anything.

```
# The address on the local server to listen to.
LMKTFY_API_ADDRESS="--address=0.0.0.0"

# Comma separated list of nodes in the etcd cluster
LMKTFY_ETCD_SERVERS="--etcd_servers=http://fed-master:4001"

# Address range to use for services
LMKTFY_SERVICE_ADDRESSES="--portal_net=10.254.0.0/16"

# Add you own!
LMKTFY_API_ARGS=""
```

* Start the appropriate services on master:

```
for SERVICES in etcd lmktfy-apiserver lmktfy-controller-manager lmktfy-scheduler; do
	systemctl restart $SERVICES
	systemctl enable $SERVICES
	systemctl status $SERVICES
done
```

* Addition of nodes:

* Create following node.json file on lmktfy master node:

```json
{
  "id": "fed-node",
  "kind": "Minion",
  "apiVersion": "v1beta1",
  "labels": {
    "name": "fed-node-label"
  }
}
```

Now create a node object internally in your lmktfy cluster by running:

```
$ lmktfyctl create -f node.json

$ lmktfyctl get nodes
NAME                LABELS              STATUS
fed-node           name=fed-node-label     Unknown

```

Please note that in the above, it only creates a representation for the node
_fed-node_ internally. It does not provision the actual _fed-node_. Also, it
is assumed that _fed-node_ (as specified in `id`) can be resolved and is
reachable from lmktfy master node. This guide will discuss how to provision
a lmktfy node (fed-node) below.

**Configure the lmktfy services on the node.**

***We need to configure the lmktfylet on the node.***

* Edit /etc/lmktfy/lmktfylet to appear as such:

```
###
# lmktfy lmktfylet (node) config

# The address for the info server to serve on (set to 0.0.0.0 or "" for all interfaces)
LMKTFYLET_ADDRESS="--address=0.0.0.0"

# You may leave this blank to use the actual hostname
LMKTFYLET_HOSTNAME="--hostname_override=fed-node"

# location of the api-server
LMKTFYLET_API_SERVER="--api_servers=http://fed-master:8080"

# Add your own!
#LMKTFYLET_ARGS=""
```

* Start the appropriate services on the node (fed-node).

```
for SERVICES in lmktfy-proxy lmktfylet docker; do 
    systemctl restart $SERVICES
    systemctl enable $SERVICES
    systemctl status $SERVICES 
done
```

* Check to make sure now the cluster can see the fed-node on fed-master, and its status changes to _Ready_.

```
lmktfyctl get nodes
NAME                LABELS              STATUS
fed-node          name=fed-node-label     Ready
```
* Deletion of nodes:

To delete _fed-node_ from your lmktfy cluster, one should run the following on fed-master (Please do not do it, it is just for information):

```
$ lmktfyctl delete -f node.json
```

*You should be finished!*

**The cluster should be running! Launch a test pod.**

You should have a functional cluster, check out [101](https://github.com/GoogleCloudPlatform/lmktfy/blob/master/examples/walkthrough/README.md)!
