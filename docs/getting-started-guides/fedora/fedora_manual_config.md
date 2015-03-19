##Getting started on [Fedora](http://fedoraproject.org)

This is a getting started guide for Fedora.  It is a manual configuration so you understand all the underlying packages / services / ports, etc...

This guide will only get ONE minion working.  Multiple minions require a functional [networking configuration](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/networking.md) done outside of kubernetes.  Although the additional kubernetes configuration requirements should be obvious.

The kubernetes package provides a few services: kube-apiserver, kube-scheduler, kube-controller-manager, kubelet, kube-proxy.  These services are managed by systemd and the configuration resides in a central location: /etc/kubernetes.  We will break the services up between the hosts.  The first host, fed-master, will be the kubernetes master.  This host will run the kube-apiserver, kube-controller-manager, and kube-scheduler.  In addition, the master will also run _etcd_ (not needed if _etcd_ runs on a different host but this guide assumes that _etcd_ and kubernetes master run on the same host).  The remaining host, fed-minion will be the minion and run kubelet, proxy and docker.

**System Information:**

Hosts:
```
fed-master = 192.168.121.9
fed-minion = 192.168.121.65
```

**Prepare the hosts:**
    
* Install kubernetes on all hosts - fed-{master,minion}.  This will also pull in etcd and docker.  This guide has been tested with kubernetes-0.12.0 but should work with later versions too.

```
yum -y install --enablerepo=updates-testing kubernetes
```

* Add master and minion to /etc/hosts on all machines (not needed if hostnames already in DNS). Make sure that communication works between fed-master and fed-minion by using a utility such as ping.

```
echo "192.168.121.9	fed-master
192.168.121.65	fed-minion" >> /etc/hosts
```

* Edit /etc/kubernetes/config which will be the same on all hosts to contain:

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

* Disable the firewall on both the master and minion, as docker does not play well with other firewall rule managers.  Please note that iptables-services does not exist on default fedora server install.

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
KUBE_ETCD_SERVERS="--etcd_servers=http://fed-master:4001"

# Address range to use for services
KUBE_SERVICE_ADDRESSES="--portal_net=10.254.0.0/16"

# Add you own!
KUBE_API_ARGS=""
```

* Edit /etc/kubernetes/controller-manager to appear as such:
```
# The following values are used to configure the kubernetes controller-manager

# defaults from config and apiserver should be adequate

# Comma separated list of minions
KUBELET_ADDRESSES="--machines=fed-minion"

# Add you own!
KUBE_CONTROLLER_MANAGER_ARGS=""
```

* Start the appropriate services on master:

```
for SERVICES in etcd kube-apiserver kube-controller-manager kube-scheduler; do 
	systemctl restart $SERVICES
	systemctl enable $SERVICES
	systemctl status $SERVICES 
done
```

**Configure the kubernetes services on the minion.**

***We need to configure the kubelet and proxy and start them.***

* Edit /etc/kubernetes/kubelet to appear as such:

```
###
# kubernetes kubelet (minion) config

# The address for the info server to serve on (set to 0.0.0.0 or "" for all interfaces)
KUBELET_ADDRESS="--address=0.0.0.0"

# You may leave this blank to use the actual hostname
KUBELET_HOSTNAME="--hostname_override=fed-minion"

# location of the api-server
KUBELET_API_SERVER="--api_servers=http://fed-master:8080"

# Add your own!
#KUBELET_ARGS=""
```

* Edit /etc/kubernetes/proxy to appear as such:

```
###
# kubernetes proxy config

# default config should be adequate

# Add your own!
KUBE_PROXY_ARGS="--master=http://fed-master:8080"
```

* Start the appropriate services on minion (fed-minion).

```
for SERVICES in kube-proxy kubelet docker; do 
    systemctl restart $SERVICES
    systemctl enable $SERVICES
    systemctl status $SERVICES 
done
```

*You should be finished!*

* Check to make sure the cluster can see the minion (on fed-master).

```
kubectl get minions
NAME                LABELS              STATUS
fed-minion          <none>              Ready
```

**The cluster should be running! Launch a test pod.**

You should have a functional cluster, check out [101](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/walkthrough/README.md)!
