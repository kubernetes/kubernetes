##Getting started on [Fedora](http://fedoraproject.org)

This is a getting started guide for Fedora.  It is a manual configuration so you understand all the underlying packages / services / ports, etc...

This guide will only get ONE minion working.  Multiple minions requires a functional [networking configuration](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/networking.md) done outside of kubernetes.  Although the additional kubernetes configuration requirements should be obvious.

The kubernetes package provides a few services: kube-apiserver, kube-scheduler, kube-controller-manager, kubelet, kube-proxy.  These services are managed by systemd and the configuration resides in a central location: /etc/kubernetes. We will break the services up between the hosts.  The first host, fed-master, will be the kubernetes master.  This host will run the kube-apiserver, kube-controller-manager, and kube-scheduler.  In addition, the master will also run _etcd_.  The remaining host, fed-minion will be the minion and run kubelet, proxy, cadvisor and docker.

**System Information:**

Hosts:
```
fed-master = 192.168.121.9
fed-minion = 192.168.121.65
```

**Prepare the hosts:**
    
* Install kubernetes on all hosts - fed-{master,minion}.  This will also pull in etcd, docker, and cadvisor.

```
yum -y install --enablerepo=updates-testing kubernetes
```

* Add master and minion to /etc/hosts on all machines (not needed if hostnames already in DNS)

```
echo "192.168.121.9	fed-master
192.168.121.65	fed-minion" >> /etc/hosts
```

* Edit /etc/kubernetes/config which will be the same on all hosts to contain:

```
# Comma seperated list of nodes in the etcd cluster
KUBE_ETCD_SERVERS="--etcd_servers=http://fed-master:4001"

# logging to stderr means we get it in the systemd journal
KUBE_LOGTOSTDERR="--logtostderr=true"

# journal message level, 0 is debug
KUBE_LOG_LEVEL="--v=0"

# Should this cluster be allowed to run privleged docker containers
KUBE_ALLOW_PRIV="--allow_privileged=false"
```

* Disable the firewall on both the master and minon, as docker does not play well with other firewall rule managers

```
systemctl disable iptables-services firewalld
systemctl stop iptables-services firewalld
```

**Configure the kubernetes services on the master.**

* Edit /etc/kubernetes/apiserver to appear as such:

```       
# The address on the local server to listen to.
KUBE_API_ADDRESS="--address=0.0.0.0"

# The port on the local server to listen on.
KUBE_API_PORT="--port=8080"

# How the replication controller and scheduler find the kube-apiserver
KUBE_MASTER="--master=http://fed-master:8080"

# Port minions listen on
KUBELET_PORT="--kubelet_port=10250"

# Address range to use for services
KUBE_SERVICE_ADDRESSES="--portal_net=10.254.0.0/16"

# Add you own!
KUBE_API_ARGS=""
```

* Edit /etc/kubernetes/controller-manager to appear as such:
```
# Comma seperated list of minions
KUBELET_ADDRESSES="--machines=fed-minion"
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

***We need to configure the kubelet and start the kubelet and proxy***

* Edit /etc/kubernetes/kubelet to appear as such:

```       
# The address for the info server to serve on
KUBELET_ADDRESS="--address=0.0.0.0"

# The port for the info server to serve on
KUBELET_PORT="--port=10250"

# You may leave this blank to use the actual hostname
KUBELET_HOSTNAME="--hostname_override=fed-minion"

# Add your own!
KUBELET_ARGS=""
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

* Check to make sure the cluster can see the minion (on fed-master)

```
kubectl get minions
```

**The cluster should be running! Launch a test pod.**

You should have a functional cluster, check out [101](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/walkthrough/README.md)!
