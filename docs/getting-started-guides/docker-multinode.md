### Running Multi-Node Kubernetes Using Docker

_Note_: These instructions are somewhat significantly more advanced than the [single node](docker.md) instructions.  If you are
interested in just starting to explore Kubernetes, we recommend that you start there.


## Master Node
We'll begin by setting up the master node.  For the purposes of illustration, we'll assume that the IP of this machine is MASTER_IP

### Setup Docker-Bootstrap
We're going to use ```flannel``` to set up networking between Docker daemons.  Flannel itself (and etcd on which it relies) will run inside of
Docker containers themselves.  To achieve this, we need a separate "bootstrap" instance of the Docker daemon.  This daemon will be started with
```--iptables=false``` so that it can only run containers with ```--net=host```.  That's sufficient to bootstrap our system.

Run:
```sh
sudo docker -d -H unix:///var/run/docker-bootstrap.sock -p /var/run/docker-bootstrap.pid --iptables=false >> /var/log/docker-bootstrap.log &&
```

### Startup etcd for flannel to use
Run:
```
docker -H unix:///var/run/docker-bootstrap.sock run --net=host -d kubernetes/etcd:2.0.5.1 /usr/local/bin/etcd --addr=127.0.0.1:4001 --bind-addr=0.0.0.0:4001 --data-dir=/var/etcd/data
```

Next,, you need to set a CIDR range for flannel.  This CIDR should be chosen to be non-overlapping with any existing network you are using:

```sh
docker -H unix:///var/run/docker-bootstrap.sock run --net=host kubernetes/etcd:2.0.5.1 etcdctl set /coreos.com/network/config '{ "Network": "10.1.0.0/16" }'
```


### Bring down Docker
To re-configure Docker to use flannel, we need to take docker down, run flannel and then restart Docker.

Turning down Docker is system dependent, it may be:

```sh
/etc/init.d/docker stop
```

or

```sh
systemctl stop docker
```

or it may be something else.

### Run flannel

Now run flanneld itself:
```sh
docker -H unix:///var/run/docker-bootstrap.sock run -d --net=host --privileged -v /dev/net:/dev/net quay.io/coreos/flannel:0.3.0
```

The previous command should have printed a really long hash, copy this hash.

Now get the subnet settings from flannel:
```
docker exec <really-long-hash-from-above-here> cat /run/flannel/subnet.env
```

### Edit the docker configuration
You now need to edit the docker configuration to activate new flags.  Again, this is system specific.

This may be in ```/etc/docker/default``` or ```/etc/systemd/service/docker.service``` or it may be elsewhere.

Regardless, you need to add the following to the docker comamnd line:
```sh
--bip=${FLANNEL_SUBNET} --mtu=${FLANNEL_MTU}
```

### Remove the existing Docker bridge
Docker creates a bridge named ```docker0``` by default.  You need to remove this:

```sh
sudo ifconfig docker0 down
sudo brctl delbr docker0
```

### Restart Docker
Again this is system dependent, it may be:

```sh
sudo /etc/init.d/docker start
```

it may be:
```sh
systemctl start docker
```

### Starting the Kubernetes Master
Ok, now that your networking is set up, you can startup Kubernetes, this is the same as the single-node case:

```sh
docker run --net=host -d -v /var/run/docker.sock:/var/run/docker.sock  gcr.io/google_containers/hyperkube:v0.14.1 /hyperkube kubelet --api_servers=http://localhost:8080 --v=2 --address=0.0.0.0 --enable_server --hostname_override=127.0.0.1 --config=/etc/kubernetes/manifests
```

### Also run the service proxy
```sh
docker run -d --net=host --privileged gcr.io/google_containers/hyperkube:v0.14.1 /hyperkube proxy --master=http://127.0.0.1:8080 --v=2
```

### Adding a new node
