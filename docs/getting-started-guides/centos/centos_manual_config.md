 
##Getting started on [CentOS](http://centos.org)

This is a getting started guide for CentOS.  It is a manual configuration so you understand all the underlying packages / services / ports, etc...

This guide will only get ONE minion working.  Multiple minions requires a functional [networking configuration](https://github.com/GoogleCloudPlatform/lmktfy/blob/master/docs/networking.md) done outside of lmktfy.  Although the additional lmktfy configuration requirements should be obvious.

The lmktfy package provides a few services: lmktfy-apiserver, lmktfy-scheduler, lmktfy-controller-manager, lmktfylet, lmktfy-proxy.  These services are managed by systemd and the configuration resides in a central location: /etc/lmktfy. We will break the services up between the hosts.  The first host, centos-master, will be the lmktfy master.  This host will run the lmktfy-apiserver, lmktfy-controller-manager, and lmktfy-scheduler.  In addition, the master will also run _etcd_.  The remaining host, centos-minion will be the minion and run lmktfylet, proxy, cadvisor and docker.

**System Information:**

Hosts:
```
centos-master = 192.168.121.9
centos-minion = 192.168.121.65
```

**Prepare the hosts:**
    
* Create virt7-testing repo on all hosts - centos-{master,minion} with following information.

```
[virt7-testing]
name=virt7-testing
baseurl=http://cbs.centos.org/repos/virt7-testing/x86_64/os/
gpgcheck=0
```

* Install lmktfy on all hosts - centos-{master,minion}.  This will also pull in etcd, docker, and cadvisor.

```
yum -y install --enablerepo=virt7-testing lmktfy
```

* Note * Using etcd-0.4.6-7 (This is temperory update in documentation)

If you do not get etcd-0.4.6-7 installed with virt7-testing repo,

In the current virt7-testing repo, the etcd package is updated which causes service failure. To avoid this,

```
yum erase etcd
```

It will uninstall the current available etcd package

```
yum install http://cbs.centos.org/kojifiles/packages/etcd/0.4.6/7.el7.centos/x86_64/etcd-0.4.6-7.el7.centos.x86_64.rpm
yum -y install --enablerepo=virt7-testing lmktfy
```

* Add master and minion to /etc/hosts on all machines (not needed if hostnames already in DNS)

```
echo "192.168.121.9	centos-master
192.168.121.65	centos-minion" >> /etc/hosts
```

* Edit /etc/lmktfy/config which will be the same on all hosts to contain:

```
# Comma separated list of nodes in the etcd cluster
LMKTFY_ETCD_SERVERS="--etcd_servers=http://centos-master:4001"

# logging to stderr means we get it in the systemd journal
LMKTFY_LOGTOSTDERR="--logtostderr=true"

# journal message level, 0 is debug
LMKTFY_LOG_LEVEL="--v=0"

# Should this cluster be allowed to run privileged docker containers
LMKTFY_ALLOW_PRIV="--allow_privileged=false"
```

* Disable the firewall on both the master and minon, as docker does not play well with other firewall rule managers

```
systemctl disable iptables-services firewalld
systemctl stop iptables-services firewalld
```

**Configure the lmktfy services on the master.**

* Edit /etc/lmktfy/apiserver to appear as such:

```       
# The address on the local server to listen to.
LMKTFY_API_ADDRESS="--address=0.0.0.0"

# The port on the local server to listen on.
LMKTFY_API_PORT="--port=8080"

# How the replication controller and scheduler find the lmktfy-apiserver
LMKTFY_MASTER="--master=http://centos-master:8080"

# Port minions listen on
LMKTFYLET_PORT="--lmktfylet_port=10250"

# Address range to use for services
LMKTFY_SERVICE_ADDRESSES="--portal_net=10.254.0.0/16"

# Add you own!
LMKTFY_API_ARGS=""
```

* Edit /etc/lmktfy/controller-manager to appear as such:
```
# Comma separated list of minions
LMKTFYLET_ADDRESSES="--machines=centos-minion"
```

* Start the appropriate services on master:

```
for SERVICES in etcd lmktfy-apiserver lmktfy-controller-manager lmktfy-scheduler; do 
	systemctl restart $SERVICES
	systemctl enable $SERVICES
	systemctl status $SERVICES 
done
```

**Configure the lmktfy services on the minion.**

***We need to configure the lmktfylet and start the lmktfylet and proxy***

* Edit /etc/lmktfy/lmktfylet to appear as such:

```       
# The address for the info server to serve on
LMKTFYLET_ADDRESS="--address=0.0.0.0"

# The port for the info server to serve on
LMKTFYLET_PORT="--port=10250"

# You may leave this blank to use the actual hostname
LMKTFYLET_HOSTNAME="--hostname_override=centos-minion"

# Add your own!
LMKTFYLET_ARGS=""
```       

* Start the appropriate services on minion (centos-minion).

```
for SERVICES in lmktfy-proxy lmktfylet docker; do 
    systemctl restart $SERVICES
    systemctl enable $SERVICES
    systemctl status $SERVICES 
done
```

*You should be finished!*

* Check to make sure the cluster can see the minion (on centos-master)

```
lmktfyctl get minions
NAME                   LABELS            STATUS
centos-minion          <none>            Ready
```

**The cluster should be running! Launch a test pod.**

You should have a functional cluster, check out [101](https://github.com/GoogleCloudPlatform/lmktfy/blob/master/examples/walkthrough/README.md)!
