##Getting started on [Fedora](http://fedoraproject.org)

This is a getting started guide for Fedora.  It is a manual configuration so you understand all the underlying packages / services / ports, etc...

This guide will only get ONE node (previously minion) working.  Multiple nodes require a functional [networking configuration](http://docs.k8s.io/networking.md) done outside of kubernetes.  Although the additional kubernetes configuration requirements should be obvious.

The kubernetes package provides a few services: kube-apiserver, kube-scheduler, kube-controller-manager, kubelet, kube-proxy.  These services are managed by systemd and the configuration resides in a central location: /etc/kubernetes.  We will break the services up between the hosts.  The first host, fed-master, will be the kubernetes master.  This host will run the kube-apiserver, kube-controller-manager, and kube-scheduler.  In addition, the master will also run _etcd_ (not needed if _etcd_ runs on a different host but this guide assumes that _etcd_ and kubernetes master run on the same host).  The remaining host, fed-node will be the node and run kubelet, proxy and docker.

**System Information:**

Hosts:
```
fed-master = 192.168.121.9
fed-node = 192.168.121.65
```

**Prepare the hosts:**
    
* Install kubernetes on all hosts - fed-{master,node}.  This will also pull in docker. Also install etcd on fed-master.  This guide has been tested with kubernetes-0.15.0 but should work with other versions too.
* The [--enablerepo=update-testing](https://fedoraproject.org/wiki/QA:Updates_Testing) directive in the yum command below will ensure that the most recent Kubernetes version that is scheduled for pre-release will be installed. This should be a more recent version than the Fedora "stable" release for Kubernetes that you would get without adding the directive. 
* If you want the very latest Kubernetes release [you can download and yum install the RPM directly from Fedora Koji](http://koji.fedoraproject.org/koji/packageinfo?packageID=19202) instead of using the yum install command below.

```
yum -y install --enablerepo=updates-testing kubernetes
```
* Install etcd and iptables

```
yum -y install etcd iptables
```

* Add master and node to /etc/hosts on all machines (not needed if hostnames already in DNS). Make sure that communication works between fed-master and fed-node by using a utility such as ping.

```
echo "192.168.121.9	fed-master
192.168.121.65	fed-node" >> /etc/hosts
```

* Edit /etc/kubernetes/config which will be the same on all hosts (master and node) to contain:

```
# Comma separated list of nodes in the etcd cluster
KUBE_MASTER="--master=http://fed-master:8080"

# logging to stderr means we get it in the systemd journal
KUBE_LOGTOSTDERR="--logtostderr=true"

# journal message level, 0 is debug
KUBE_LOG_LEVEL="--v=0"

# Should this cluster be allowed to run privileged docker containers
KUBE_ALLOW_PRIV="--allow_privileged=false"
```

* Disable the firewall on both the master and node, as docker does not play well with other firewall rule managers.  Please note that iptables-services does not exist on default fedora server install.

```
systemctl disable iptables-services firewalld
systemctl stop iptables-services firewalld
```

**Configure the kubernetes services on the master.**

* Edit /etc/kubernetes/apiserver to appear as such.  The portal_net IP addresses must be an unused block of addresses, not used anywhere else.  They do not need to be routed or assigned to anything.

```
# The address on the local server to listen to.
KUBE_API_ADDRESS="--address=0.0.0.0"

# Comma separated list of nodes in the etcd cluster
KUBE_ETCD_SERVERS="--etcd_servers=http://127.0.0.1:4001"

# Address range to use for services
KUBE_SERVICE_ADDRESSES="--portal_net=10.254.0.0/16"

# Add your own!
KUBE_API_ARGS=""
```

* *Optional* Edit /etc/kubernetes/controller-manager and remove --machines=127.0.0.1 from the KUBELET_ADDRESSES. Leaving this in won't hurt anything but it will cause the output to note that the 127.0.0.1 node is NotReady because we will not be configuring one in this guide.

```
KUBELET_ADDRESSES=""

KUBE_CONTROLLER_MANAGER_ARGS=""
```


* Edit /etc/etcd/etcd.conf,let the etcd to listen all the ip instead of 127.0.0.1,if not ,you will get the error like "connection refused"
```
ETCD_LISTEN_CLIENT_URLS="http://0.0.0.0:4001"
```

* Start the appropriate services on master:

```
for SERVICES in etcd kube-apiserver kube-controller-manager kube-scheduler; do
	systemctl restart $SERVICES
	systemctl enable $SERVICES
	systemctl status $SERVICES
done
```

* Addition of nodes:

* Create following node.json file on kubernetes master node:

```json
{
    "apiVersion": "v1beta3",
    "kind": "Node",
    "metadata": {
        "name": "fed-node",
        "labels":{ "name": "fed-node-label"}
    },
    "spec": {
        "externalID": "fed-node"
    }
}
```

Now create a node object internally in your kubernetes cluster by running:

```
$ kubectl create -f node.json

$ kubectl get nodes
NAME                LABELS              STATUS
fed-node           name=fed-node-label     Unknown

```

Please note that in the above, it only creates a representation for the node
_fed-node_ internally. It does not provision the actual _fed-node_. Also, it
is assumed that _fed-node_ (as specified in `name`) can be resolved and is
reachable from kubernetes master node. This guide will discuss how to provision
a kubernetes node (fed-node) below.

**Configure the kubernetes services on the node.**

***We need to configure the kubelet on the node.***

* Edit /etc/kubernetes/kubelet to appear as such:

```
###
# kubernetes kubelet (node) config

# The address for the info server to serve on (set to 0.0.0.0 or "" for all interfaces)
KUBELET_ADDRESS="--address=0.0.0.0"

# You may leave this blank to use the actual hostname
KUBELET_HOSTNAME="--hostname_override=fed-node"

# location of the api-server
KUBELET_API_SERVER="--api_servers=http://fed-master:8080"

# Add your own!
#KUBELET_ARGS=""
```

* Start the appropriate services on the node (fed-node).

```
for SERVICES in kube-proxy kubelet docker; do 
    systemctl restart $SERVICES
    systemctl enable $SERVICES
    systemctl status $SERVICES 
done
```

* Check to make sure now the cluster can see the fed-node on fed-master, and its status changes to _Ready_.

```
kubectl get nodes
NAME                LABELS              STATUS
fed-node          name=fed-node-label     Ready
```
* Deletion of nodes:

To delete _fed-node_ from your kubernetes cluster, one should run the following on fed-master (Please do not do it, it is just for information):

```
$ kubectl delete -f node.json
```

*You should be finished!*

**The cluster should be running! Launch a test pod.**

You should have a functional cluster, check out [101](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/walkthrough/README.md)!
