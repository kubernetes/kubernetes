## Adding a Kubernetes worker node via Docker.

These instructions are very similar to the master set-up above, but they are duplicated for clarity.
You need to repeat these instructions for each node you want to join the cluster.
We will assume that the IP address of this node is ```${NODE_IP}``` and you have the IP address of the master in ```${MASTER_IP}``` that you created in the [master instructions](master.md).

For each worker node, there are three steps:
   * [Set up ```flanneld``` on the worker node](#set-up-flanneld-on-the-worker-node)
   * [Start kubernetes on the worker node](#start-kubernetes-on-the-worker-node)
   * [Add the worker to the cluster](#add-the-node-to-the-cluster)

### Set up Flanneld on the worker node
As before, the Flannel daemon is going to provide network connectivity.

#### Set up a bootstrap docker:
As previously, we need a second instance of the Docker daemon running to bootstrap the flannel networking.

Run:
```sh
sudo sh -c 'docker -d -H unix:///var/run/docker-bootstrap.sock -p /var/run/docker-bootstrap.pid --iptables=false --ip-masq=false --bridge=none --graph=/var/lib/docker-bootstrap 2> /var/log/docker-bootstrap.log 1> /dev/null &'
```

_Important Note_:
If you are running this on a long running system, rather than experimenting, you should run the bootstrap Docker instance under something like SysV init, upstart or systemd so that it is restarted
across reboots and failures.

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

Now run flanneld itself, this call is slightly different from the above, since we point it at the etcd instance on the master.
```sh
sudo docker -H unix:///var/run/docker-bootstrap.sock run -d --net=host --privileged -v /dev/net:/dev/net quay.io/coreos/flannel:0.3.0 /opt/bin/flanneld --etcd-endpoints=http://${MASTER_IP}:4001
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

### Start Kubernetes on the worker node
#### Run the kubelet
Again this is similar to the above, but the ```--api_servers``` now points to the master we set up in the beginning.

```sh
sudo docker run --net=host -d -v /var/run/docker.sock:/var/run/docker.sock  gcr.io/google_containers/hyperkube:v0.14.2 /hyperkube kubelet --api_servers=http://${MASTER_IP}:8080 --v=2 --address=0.0.0.0 --enable_server --hostname_override=$(hostname -i)
```

#### Run the service proxy
The service proxy provides load-balancing between groups of containers defined by Kubernetes ```Services```

```sh
sudo docker run -d --net=host --privileged gcr.io/google_containers/hyperkube:v0.14.2 /hyperkube proxy --master=http://${MASTER_IP}:8080 --v=2
```


### Add the node to the cluster

On the master you created above, create a file named ```node.yaml``` make it's contents:

```yaml
apiVersion: v1beta1
externalID: ${NODE_IP}
hostIP: ${NODE_IP}
id: ${NODE_IP}
kind: Node
resources:
  capacity:
    # Adjust these to match your node
    cpu: "1"
    memory: 3892043776
```

Make the API call to add the node, you should do this on the master node that you created above.  Otherwise you need to add ```-s=http://${MASTER_IP}:8080``` to point ```kubectl``` at the master.

```sh
./kubectl create -f node.yaml
```

### Next steps

Move on to [testing your cluster](testing.md) or [add another node](#adding-a-kubernetes-worker-node-via-docker)
