##Getting started on [Fedora](http://fedoraproject.org)

This is a getting started guide for Fedora.  It is a manual configuration so you understand all the underlying packages / services / ports, etc...

This guide will only get ONE minion working.  Multiple minions requires a functional [networking configuration](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/networking.md) done outside of kubernetes.  Although the additional kubernetes configuration requirements should be obvious.

The guide is broken into 3 sections:

1. Prepare the hosts.
2. Configuring the two hosts, a master and a minion.
3. Basic functionality test.

The kubernetes package provides a few services: apiserver, scheduler, controller, kubelet, proxy.  These services are managed by systemd and the configuration resides in a central location: /etc/kubernetes. We will break the services up between the hosts.  The first host, fed-master, will be the kubernetes master.  This host will run the apiserver, controller, and scheduler.  In addition, the master will also run _etcd_.  The remaining host, fed-minion will be the minion and run kubelet, proxy, cadvisor and docker.

**System Information:**

Hosts:
```
fed-master = 192.168.121.9
fed-minion = 192.168.121.65
```

Versions:

```
Fedora release 20 (Heisenbug)

etcd-0.4.6-6.fc20.x86_64
kubernetes-0.4-0.2.gitd5377e4.fc22.x86_64
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

* Edit /etc/kubernetes/config which will be the same on all hosts

```
###
# kubernetes system config
#
# The following values are used to configure various aspects of all
# kubernetes services, including
#
#   kubernetes-apiserver.service
#   kubernetes-controller-manager.service
#   kubernetes-scheduler.service
#   kubelet.service
#   kubernetes-proxy.service

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

***For this you need to configure the apiserver. The apiserver, controller-manager, and scheduler along with the etcd, will need to be started***

* Edit /etc/kubernetes/apiserver to appear as such:

```       
###
# kubernetes system config
#
# The following values are used to configure the kubernetes-apiserver
#

# The address on the local server to listen to.
KUBE_API_ADDRESS="--address=0.0.0.0"

# The port on the local server to listen on.
KUBE_API_PORT="--port=8080"

# How the replication controller and scheduler find the apiserver
KUBE_MASTER="--master=fed-master:8080"

# Port minions listen on
KUBELET_PORT="--kubelet_port=10250"

# Address range to use for services
KUBE_SERVICE_ADDRESSES="--portal_net=10.254.0.0/16"

# Add you own!
KUBE_API_ARGS=""
```

* Edit /etc/kubernetes/controller-manager to appear as such:
```
###
# kubernetes system config
#
# The following values are used to configure the kubernetes-controller-manager
#

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

* Take a look at what ports the services are running on.

```       
# netstat -tulnp | grep -E "(kube)|(etcd)"
```       

* Test etcd on the master (fed-master)

```       
curl -s -L http://fed-master:4001/version
curl -s -L http://fed-master:4001/v2/keys/mykey -XPUT -d value="this is awesome" | python -mjson.tool
curl -s -L http://fed-master:4001/v2/keys/mykey | python -mjson.tool
curl -s -L http://fed-master:4001/v2/keys/mykey -XDELETE | python -mjson.tool
```       

* Poke the apiserver just a bit
```
curl -s -L http://fed-master:8080/version | python -mjson.tool
curl -s -L http://fed-master:8080/api/v1beta1/pods | python -mjson.tool
curl -s -L http://fed-master:8080/api/v1beta1/minions | python -mjson.tool
curl -s -L http://fed-master:8080/api/v1beta1/services | python -mjson.tool
```
**Configure the kubernetes services on the minion.**

***We need to configure the kubelet and start the kubelet and proxy***

* Edit /etc/kubernetes/kubelet to appear as such:

```       
###
# kubernetes kubelet (minion) config

# The address for the info server to serve on
KUBELET_ADDRESS="--address=fed-minion"

# The port for the info server to serve on
KUBELET_PORT="--port=10250"

# You may leave this blank to use the actual hostname
KUBELET_HOSTNAME="--hostname_override=fed-minion"

# Add your won!
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

* Take a look at what ports the services are running on.

```       
netstat -tulnp | grep -E "(kube)|(docker)|(cadvisor)"
```       

* Check to make sure the cluster can see the minion (on fed-master)

```
kubectl get minions
```

**The cluster should be running! Launch a test pod.**

* Create a file on fed-master called apache.json that looks as such:

```
{
    "apiVersion": "v1beta1",
    "kind": "Pod",
    "id": "apache",
    "namespace": "default",
    "labels": {
        "name": "apache"
    },
    "desiredState": {
        "manifest": {
            "version": "v1beta1",
            "id": "apache",
            "volumes": null,
            "containers": [
                {
                    "name": "master",
                    "image": "fedora/apache",
                    "ports": [
                        {
                            "containerPort": 80,
                            "hostPort": 80,
                            "protocol": "TCP"
                        }
                    ],
                }
            ],
            "restartPolicy": {
                "always": {}
            }
        },
    },
}
```       

This json file is describing the attributes of the application environment.  For example, it is giving it a "kind", "id", "name", "ports", and "image".  Since the fedora/apache images doesn't exist in our environment yet, it will be pulled down automatically as part of the deployment process.

For more information about which options can go in the schema, check out the docs on the kubernetes github page.

* Deploy the fedora/apache image via the apache.json file.

```       
kubectl create -f apache.json
```
       

* You can monitor progress of the operations with these commands:
On the master (fed-master) -

```       
journalctl -f -l -xn -u kube-apiserver -u etcd -u kube-scheduler
```

* On the minion (fed-minion) -

```       
journalctl -f -l -xn -u kubelet -u kube-proxy -u docker
```
       

* After the pod is deployed, you can also list the pod.

```       
# /usr/bin/kubectl get pods 
ID                  IMAGE(S)            HOST                LABELS              STATUS
apache              fedora/apache       192.168.121.65/     name=apache         Running
```       

The state might be 'Waiting'.  This indicates that docker is still attempting to download and launch the container.

* You can get even more information about the pod like this.

```       
kubectl get --output=json pods/apache | python -mjson.tool
```       

* Finally, on the minion (fed-minion), check that the service is available, running, and functioning.

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
# /usr/bin/kubectl --server=http://fed-master:8080 delete pod apache
```       

Of course this just scratches the surface. I recommend you head off to the kubernetes github page and follow the guestbook example.  It's a bit more complicated but should expose you to more functionality.

You can play around with other Fedora images by building from Fedora Dockerfiles.
