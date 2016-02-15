<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/docker-multinode/master.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Installing a Kubernetes Master Node via Docker

We'll begin by setting up the master node.  For the purposes of illustration, we'll assume that the IP of this machine
is `${MASTER_IP}`.  We'll need to run several versioned Kubernetes components, so we'll assume that the version we want
to run is `${K8S_VERSION}`, which should hold a released version of Kubernetes >= "1.2.0-alpha.7"

Enviroinment variables used:

```sh
export MASTER_IP=<the_master_ip_here>
export K8S_VERSION=<your_k8s_version (e.g. 1.2.0-alpha.7)>
export ETCD_VERSION=<your_etcd_version (e.g. 2.2.1)>
export FLANNEL_VERSION=<your_flannel_version (e.g. 0.5.5)>
export FLANNEL_IFACE=<flannel_interface (defaults to eth0)>
export FLANNEL_IPMASQ=<flannel_ipmasq_flag (defaults to true)>
```

There are two main phases to installing the master:
   * [Setting up `flanneld` and `etcd`](#setting-up-flanneld-and-etcd)
   * [Starting the Kubernetes master components](#starting-the-kubernetes-master)


## Setting up flanneld and etcd

_Note_:
This guide expects **Docker 1.7.1 or higher**.

### Setup Docker Bootstrap

We're going to use `flannel` to set up networking between Docker daemons.  Flannel itself (and etcd on which it relies) will run inside of
Docker containers themselves.  To achieve this, we need a separate "bootstrap" instance of the Docker daemon.  This daemon will be started with
`--iptables=false` so that it can only run containers with `--net=host`.  That's sufficient to bootstrap our system.

Run:

```sh
sudo sh -c 'docker -d -H unix:///var/run/docker-bootstrap.sock -p /var/run/docker-bootstrap.pid --iptables=false --ip-masq=false --bridge=none --graph=/var/lib/docker-bootstrap 2> /var/log/docker-bootstrap.log 1> /dev/null &'
```

_Important Note_:
If you are running this on a long running system, rather than experimenting, you should run the bootstrap Docker instance under something like SysV init, upstart or systemd so that it is restarted
across reboots and failures.


### Startup etcd for flannel and the API server to use

Run:

```sh
sudo docker -H unix:///var/run/docker-bootstrap.sock run -d \
    --net=host \
    gcr.io/google_containers/etcd-amd64:${ETCD_VERSION} \
    /usr/local/bin/etcd \
        --listen-client-urls=http://127.0.0.1:4001,http://${MASTER_IP}:4001 \
        --advertise-client-urls=http://${MASTER_IP}:4001 \
        --data-dir=/var/etcd/data
```

Next, you need to set a CIDR range for flannel.  This CIDR should be chosen to be non-overlapping with any existing network you are using:

```sh
sudo docker -H unix:///var/run/docker-bootstrap.sock run \
    --net=host \
    gcr.io/google_containers/etcd-amd64:${ETCD_VERSION} \
    etcdctl set /coreos.com/network/config '{ "Network": "10.1.0.0/16" }'
```


### Set up Flannel on the master node

Flannel is a network abstraction layer build by CoreOS, we will use it to provide simplified networking between our Pods of containers.

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

or

```sh
sudo service docker stop
```

or it may be something else.

#### Run flannel

Now run flanneld itself:

```sh
sudo docker -H unix:///var/run/docker-bootstrap.sock run -d \
    --net=host \
    --privileged \
    -v /dev/net:/dev/net \
    quay.io/coreos/flannel:${FLANNEL_VERSION} \
        --ip-masq=${FLANNEL_IPMASQ} \
        --iface=${FLANNEL_IFACE}
```

The previous command should have printed a really long hash, the container id, copy this hash.

Now get the subnet settings from flannel:

```sh
sudo docker -H unix:///var/run/docker-bootstrap.sock exec <really-long-hash-from-above-here> cat /run/flannel/subnet.env
```

#### Edit the docker configuration

You now need to edit the docker configuration to activate new flags.  Again, this is system specific.

This may be in `/etc/default/docker` or `/etc/systemd/service/docker.service` or it may be elsewhere.

Regardless, you need to add the following to the docker command line:

```sh
--bip=${FLANNEL_SUBNET} --mtu=${FLANNEL_MTU}
```

#### Remove the existing Docker bridge

Docker creates a bridge named `docker0` by default.  You need to remove this:

```sh
sudo /sbin/ifconfig docker0 down
sudo brctl delbr docker0
```

You may need to install the `bridge-utils` package for the `brctl` binary.

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
sudo docker run \
    --volume=/:/rootfs:ro \
    --volume=/sys:/sys:ro \
    --volume=/var/lib/docker/:/var/lib/docker:rw \
    --volume=/var/lib/kubelet/:/var/lib/kubelet:rw \
    --volume=/var/run:/var/run:rw \
    --net=host \
    --privileged=true \
    --pid=host \
    -d \
    gcr.io/google_containers/hyperkube-amd64:v${K8S_VERSION} \
    /hyperkube kubelet \
        --allow-privileged=true \
        --api-servers=http://localhost:8080 \
        --v=2 \
        --address=0.0.0.0 \
        --enable-server \
        --hostname-override=127.0.0.1 \
        --config=/etc/kubernetes/manifests-multi \
        --containerized \
        --cluster-dns=10.0.0.10 \
        --cluster-domain=cluster.local
```

> Note that `--cluster-dns` and `--cluster-domain` is used to deploy dns, feel free to discard them if dns is not needed.

### Test it out

At this point, you should have a functioning 1-node cluster.  Let's test it out!

Download the kubectl binary for `${K8S_VERSION}` (look at the URL in the following links) and make it available by editing your PATH environment variable.
([OS X/amd64](http://storage.googleapis.com/kubernetes-release/release/v1.2.0-alpha.7/bin/darwin/amd64/kubectl))
([OS X/386](http://storage.googleapis.com/kubernetes-release/release/v1.2.0-alpha.7/bin/darwin/386/kubectl))
([linux/amd64](http://storage.googleapis.com/kubernetes-release/release/v1.2.0-alpha.7/bin/linux/amd64/kubectl))
([linux/386](http://storage.googleapis.com/kubernetes-release/release/v1.2.0-alpha.7/bin/linux/386/kubectl))
([linux/arm](http://storage.googleapis.com/kubernetes-release/release/v1.2.0-alpha.7/bin/linux/arm/kubectl))

For example, OS X:

```console
$ wget http://storage.googleapis.com/kubernetes-release/release/v${K8S_VERSION}/bin/darwin/amd64/kubectl
$ chmod 755 kubectl
$ PATH=$PATH:`pwd`
```

Linux:

```console
$ wget http://storage.googleapis.com/kubernetes-release/release/v${K8S_VERSION}/bin/linux/amd64/kubectl
$ chmod 755 kubectl
$ PATH=$PATH:`pwd`
```

Now you can list the nodes:

```sh
kubectl get nodes
```

This should print something like:

```console
NAME        LABELS                             STATUS
127.0.0.1   kubernetes.io/hostname=127.0.0.1   Ready
```

If the status of the node is `NotReady` or `Unknown` please check that all of the containers you created are successfully running.
If all else fails, ask questions on [Slack](../../troubleshooting.md#slack).


### Next steps

Move on to [adding one or more workers](worker.md) or [deploy a dns](deployDNS.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker-multinode/master.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
