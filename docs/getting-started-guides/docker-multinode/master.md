## Installing a Kubernetes Master Node via Docker
We'll begin by setting up the master node.  For the purposes of illustration, we'll assume that the IP of this machine is ```${MASTER_IP}```

There are two main phases to installing the master:
   * [Setting up ```flanneld``` and ```etcd```](#setting-up-flanneld-and-etcd)
   * [Starting the Kubernetes master components](#starting-the-kubernetes-master)


## Setting up flanneld and etcd

### Setup Docker-Bootstrap
We're going to use ```flannel``` to set up networking between Docker daemons.  Flannel itself (and etcd on which it relies) will run inside of
Docker containers themselves.  To achieve this, we need a separate "bootstrap" instance of the Docker daemon.  This daemon will be started with
```--iptables=false``` so that it can only run containers with ```--net=host```.  That's sufficient to bootstrap our system.

Run:
```sh
sudo sh -c 'docker -d -H unix:///var/run/docker-bootstrap.sock -p /var/run/docker-bootstrap.pid --iptables=false --ip-masq=false --bridge=none --graph=/var/lib/docker-bootstrap 2> /var/log/docker-bootstrap.log 1> /dev/null &'
```

_Important Note_:
If you are running this on a long running system, rather than experimenting, you should run the bootstrap Docker instance under something like SysV init, upstart or systemd so that it is restarted
across reboots and failures.


### Startup etcd for flannel and the API server to use
Run:
```
sudo docker -H unix:///var/run/docker-bootstrap.sock run --net=host -d kubernetes/etcd:2.0.5.1 /usr/local/bin/etcd --addr=127.0.0.1:4001 --bind-addr=0.0.0.0:4001 --data-dir=/var/etcd/data
```

Next, you need to set a CIDR range for flannel.  This CIDR should be chosen to be non-overlapping with any existing network you are using:

```sh
sudo docker -H unix:///var/run/docker-bootstrap.sock run --net=host kubernetes/etcd:2.0.5.1 etcdctl set /coreos.com/network/config '{ "Network": "10.1.0.0/16" }'
```


### Set up Flannel on the master node
Flannel is a network abstraction layer build by CoreOS, we will use it to provide simplfied networking between our Pods of containers.

Flannel re-configures the bridge that Docker uses for networking.  As a result we need to stop Docker, reconfigure its networking, and then restart Docker.

#### Bring down Docker
To re-configure Docker to use flannel, we need to take docker down, run flannel and then restart Docker.

Turning down Docker is system dependent, it may be:

```sh
sudo /etc/init.d/docker stop
```

or

```sh
sudo systemctl stop docker
```

or it may be something else.

#### Run flannel

Now run flanneld itself:
```sh
sudo docker -H unix:///var/run/docker-bootstrap.sock run -d --net=host --privileged -v /dev/net:/dev/net quay.io/coreos/flannel:0.3.0
```

The previous command should have printed a really long hash, copy this hash.

Now get the subnet settings from flannel:
```
sudo docker -H unix:///var/run/docker-bootstrap.sock exec <really-long-hash-from-above-here> cat /run/flannel/subnet.env
```

#### Edit the docker configuration
You now need to edit the docker configuration to activate new flags.  Again, this is system specific.

This may be in ```/etc/default/docker``` or ```/etc/systemd/service/docker.service``` or it may be elsewhere.

Regardless, you need to add the following to the docker comamnd line:
```sh
--bip=${FLANNEL_SUBNET} --mtu=${FLANNEL_MTU}
```

#### Remove the existing Docker bridge
Docker creates a bridge named ```docker0``` by default.  You need to remove this:

```sh
sudo /sbin/ifconfig docker0 down
sudo brctl delbr docker0
```

You may need to install the ```bridge-utils``` package for the ```brctl``` binary.

#### Restart Docker
Again this is system dependent, it may be:

```sh
sudo /etc/init.d/docker start
```

it may be:
```sh
systemctl start docker
```

## Starting the Kubernetes Master
Ok, now that your networking is set up, you can startup Kubernetes, this is the same as the single-node case, we will use the "main" instance of the Docker daemon for the Kubernetes components.

```sh
sudo docker run --net=host -d -v /var/run/docker.sock:/var/run/docker.sock  gcr.io/google_containers/hyperkube:v0.14.2 /hyperkube kubelet --api_servers=http://localhost:8080 --v=2 --address=0.0.0.0 --enable_server --hostname_override=127.0.0.1 --config=/etc/kubernetes/manifests-multi
```

### Also run the service proxy
```sh
sudo docker run -d --net=host --privileged gcr.io/google_containers/hyperkube:v0.14.2 /hyperkube proxy --master=http://127.0.0.1:8080 --v=2
```

### Test it out
At this point, you should have a functioning 1-node cluster.  Let's test it out!

Download the kubectl binary
([OS X](http://storage.googleapis.com/kubernetes-release/release/v0.14.2/bin/darwin/amd64/kubectl))
([linux](http://storage.googleapis.com/kubernetes-release/release/v0.14.2/bin/linux/amd64/kubectl))

List the nodes

```sh
kubectl get nodes
```

This should print:
```
NAME        LABELS    STATUS
127.0.0.1   <none>    Ready
```

If the status of the node is ```NotReady``` or ```Unknown``` please check that all of the containers you created are successfully running.
If all else fails, ask questions on IRC at #google-containers.


### Next steps
Move on to [adding one or more workers](worker.md)
