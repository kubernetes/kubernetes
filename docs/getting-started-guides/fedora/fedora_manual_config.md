##Getting started on [Fedora](http://fedoraproject.org)

This is a getting started guide for Fedora.  It is a manual configuration so you understand all the underlying packages / services / ports, etc...  The guide is broken into 2 sections:

1. Prepare the hosts.
2. Configuring the two hosts, a master and a minion.
3. Basic functionality test.

The kubernetes package provides a few services: apiserver, scheduler, controller, kubelet, proxy.  These services are managed by systemd and the configuration resides in a central location: /etc/kubernetes. We will break the services up between the hosts.  The first host, fed1, will be the kubernetes master.  This host will run the apiserver, controller, and scheduler.  In addition, the master will also run _etcd_.  The remaining host, fed2 will be the minion and run kubelet, proxy and docker.

**System Information:**

Hosts:
```
fed1 = 192.168.121.9
fed2 = 192.168.121.65
```

Versions:

```
Fedora release 20 (Heisenbug)

etcd-0.4.6-3.fc20.x86_64
kubernetes-0.2-0.4.gitcc7999c.fc20.x86_64
```

  
**Prepare the hosts:**
    
* Enable the copr repos on all hosts.  Colin Walters has already built the appropriate etcd / kubernetes packages for rawhide.  You can see the copr repo [here](https://copr.fedoraproject.org/coprs/walters/atomic-next/).

```
# yum -y install dnf dnf-plugins-core
# dnf copr enable walters/atomic-next
# yum repolist walters-atomic-next/x86_64
Loaded plugins: langpacks
repo id                                     repo name                                                     status
walters-atomic-next/x86_64       Copr repo for atomic-next owned by walters      37
repolist: 37
```

* Install kubernetes on all hosts - fed{1,2}.  This will also pull in etcd and cadvisor.  In addition, pull in the iptables-services package as we will not be using firewalld.

```
yum -y install kubernetes
yum -y install iptables-services
```

* Pick a host and explore the packages. 

```
rpm -qi kubernetes
rpm -qc kubernetes
rpm -ql kubernetes
rpm -ql etcd
rpm -qi etcd
rpm -qi cadvisor
rpm -qc cadvisor
rpm -ql cadvisor
```

* Install docker-io on fed2

```
# yum erase docker -y
# yum -y install docker-io
```

** Configure the kubernetes services on fed1. For this exercise, the apiserver, controller manager, iptables and etcd will be started on fed1. **

* Configure the /etc/kubernetes/apiserver to appear as such:

```       
###
# kubernetes system config
#
# The following values are used to configure the kubernetes-apiserver
#

# The address on the local server to listen to.
KUBE_API_ADDRESS="0.0.0.0"

# The port on the local server to listen on.
KUBE_API_PORT="8080"

# How the replication controller and scheduler find the apiserver
KUBE_MASTER="192.168.121.9:8080"

# Comma seperated list of minions
MINION_ADDRESSES="192.168.121.65"

# Port minions listen on
MINION_PORT="10250"
```

* Configure the /etc/kubernetes/config to appear as such:

```
###
# kubernetes system config
#
# The following values are used to configure various aspects of all
# kubernetes services, including
#
#   kubernetes-apiserver.service
#   kubernetes-controller-manager.service
#   kubernetes-kubelet.service
#   kubernetes-proxy.service

# Comma seperated list of nodes in the etcd cluster
KUBE_ETCD_SERVERS="http://192.168.121.9:4001"

# logging to stderr means we get it in the systemd journal
KUBE_LOGTOSTDERR="true"

# journal message level, 0 is debug
KUBE_LOG_LEVEL=0

KUBE_ALLOW_PRIV="true"
```

* Start the appropriate services on fed1:

```
for SERVICES in etcd kube-apiserver kube-controller-manager kube-scheduler; do 
	systemctl restart $SERVICES
	systemctl enable $SERVICES
	systemctl status $SERVICES 
done
```

* Test etcd on the master (fed1) and make sure it's working (pulled from CoreOS github page):

```       
curl -L http://127.0.0.1:4001/v2/keys/mykey -XPUT -d value="this is awesome"
curl -L http://127.0.0.1:4001/v2/keys/mykey
curl -L http://127.0.0.1:4001/version
```       

* Take a look at what ports the services are running on.

```       
# netstat -tulnp
```       

* Open up the ports for etcd and the kubernetes API server on the master (fed1).

```       
/sbin/iptables -I INPUT 1 -p tcp --dport 8080 -j ACCEPT -m comment --comment "kube-apiserver"
/sbin/iptables -I INPUT 1 -p tcp --dport 4001 -j ACCEPT -m comment --comment "etcd_client"
service iptables save
systemctl daemon-reload
systemctl restart iptables
systemctl status iptables
```       

** Configure the kubernetes services on fed2. For this exercise, the kubelet, kube-proxy, and iptables fed2. **

* Configure the /etc/kubernetes/kubelet to appear as such:

```       
###
# kubernetes kublet (minion) config

# The address for the info server to serve on
MINION_ADDRESS="192.168.121.65"

# The port for the info server to serve on
MINION_PORT="10250"

# You may leave this blank to use the actual hostname
MINION_HOSTNAME="192.168.121.65"
```       

* Configure the /etc/kubernetes/config to appear as such:

```
###
# kubernetes system config
#
# The following values are used to configure various aspects of all
# kubernetes services, including
#
#   kubernetes-apiserver.service
#   kubernetes-controller-manager.service
#   kubernetes-kubelet.service
#   kubernetes-proxy.service

# Comma seperated list of nodes in the etcd cluster
KUBE_ETCD_SERVERS="http://192.168.121.9:4001"

# logging to stderr means we get it in the systemd journal
KUBE_LOGTOSTDERR="true"

# journal message level, 0 is debug
KUBE_LOG_LEVEL=0

KUBE_ALLOW_PRIV="true"
```

* Start the appropriate services on fed2.

```
for SERVICES in kube-proxy kubelet docker; do 
    systemctl restart $SERVICES
    systemctl enable $SERVICES
    systemctl status $SERVICES 
done
```

* Take a look at what ports the services are running on.

```       
netstat -tulnp
```       

* Open up the port for the kubernetes kubelet server on the minion (fed2).

```       
/sbin/iptables -I INPUT 1 -p tcp --dport 10250 -j ACCEPT -m comment --comment "kubelet"
service iptables save
systemctl daemon-reload
systemctl restart iptables
systemctl status iptables
```       
 

* Now the two servers are set up to kick off a sample application.  In this case, we'll deploy a web server to fed2.  Start off by making a file in roots home directory on fed1 called apache.json that looks as such:

```       
cat ~/apache.json 
{
  "id": "apache",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "apache-1",
      "containers": [{
        "name": "master",
        "image": "fedora/apache",
        "ports": [{
          "containerPort": 80,
          "hostPort": 80
        }]
      }]
    }
  },
  "labels": {
    "name": "apache"
  }
}
```       

This json file is describing the attributes of the application environment.  For example, it is giving it an "id", "name", "ports", and "image".  Since the fedora/apache images doesn't exist in our environment yet, it will be pulled down automatically as part of the deployment process.  I have seen errors though where kubernetes was looking for a cached image.  In that case I did a manual "docker pull fedora/apache" and that seemed to resolve.
For more information about which options can go in the schema, check out the docs on the kubernetes github page.

* Deploy the fedora/apache image via the apache.json file.

```       
/bin/kubecfg -c apache.json create pods
```
       

* You can monitor progress of the operations with these commands:
On the master (fed1) -

```       
journalctl -f -l -xn -u kube-apiserver -u etcd -u kube-scheduler
```

* On the minion (fed2) -

```       
journalctl -f -l -xn -u kubelet.service -u kube-proxy -u docker
```
       

* This is what a successful expected result should look like:

```       
/bin/kubecfg -c apache.json create pods 
ID                  Image(s)            Host                Labels              Status
----------          ----------          ----------          ----------          ----------
apache              fedora/apache       /                   name=apache         Waiting
```

* After the pod is deployed, you can also list the pod.

```       
/bin/kubecfg -c apache.json list pods 
ID                  Image(s)            Host                Labels              Status
----------          ----------          ----------          ----------          ----------
apache              fedora/apache       192.168.121.65/     name=apache         Running
```       

* You can get even more information about the pod like this.

```       
/bin/kubecfg -json get pods/apache | python -mjson.tool
```       

* Finally, on the minion (fed2), check that the service is available, running, and functioning.

```       
docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
kubernetes/pause    latest              6c4579af347b        7 weeks ago         239.8 kB
fedora/apache       latest              6927a389deb6        3 months ago        450.6 MB

docker ps -l
CONTAINER ID        IMAGE                  COMMAND             CREATED             STATUS              PORTS               NAMES
05c69c00ea48        fedora/apache:latest   "/run-apache.sh"    2 minutes ago       Up 2 minutes                            k8s--master.3f918229--apache.etcd--8cd6efe6_-_3a95_-_11e4_-_b618_-_5254005318cb--9bb78458

curl http://localhost
Apache
```       

* To delete the container.

```       
/bin/kubecfg -h http://127.0.0.1:8080 delete /pods/apache
```       

Of course this just scratches the surface. I recommend you head off to the kubernetes github page and follow the guestbook example.  It's a bit more complicated but should expose you to more functionality.

You can play around with other Fedora images by building from Fedora Dockerfiles. Check here at Github. 

