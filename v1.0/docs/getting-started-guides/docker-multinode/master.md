---
layout: docwithnav
title: "Installing a Kubernetes Master Node via Docker"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Installing a Kubernetes Master Node via Docker

We'll begin by setting up the master node.  For the purposes of illustration, we'll assume that the IP of this machine is `${MASTER_IP}`

There are two main phases to installing the master:
   * [Setting up `flanneld` and `etcd`](#setting-up-flanneld-and-etcd)
   * [Starting the Kubernetes master components](#starting-the-kubernetes-master)


## Setting up flanneld and etcd

_Note_:
There is a [bug](https://github.com/docker/docker/issues/14106) in Docker 1.7.0 that prevents this from working correctly.
Please install Docker 1.6.2 or wait for Docker 1.7.1.

### Setup Docker-Bootstrap

We're going to use `flannel` to set up networking between Docker daemons.  Flannel itself (and etcd on which it relies) will run inside of
Docker containers themselves.  To achieve this, we need a separate "bootstrap" instance of the Docker daemon.  This daemon will be started with
`--iptables=false` so that it can only run containers with `--net=host`.  That's sufficient to bootstrap our system.

Run:

{% highlight sh %}
{% raw %}
sudo sh -c 'docker -d -H unix:///var/run/docker-bootstrap.sock -p /var/run/docker-bootstrap.pid --iptables=false --ip-masq=false --bridge=none --graph=/var/lib/docker-bootstrap 2> /var/log/docker-bootstrap.log 1> /dev/null &'
{% endraw %}
{% endhighlight %}

_Important Note_:
If you are running this on a long running system, rather than experimenting, you should run the bootstrap Docker instance under something like SysV init, upstart or systemd so that it is restarted
across reboots and failures.


### Startup etcd for flannel and the API server to use

Run:

{% highlight sh %}
{% raw %}
sudo docker -H unix:///var/run/docker-bootstrap.sock run --net=host -d gcr.io/google_containers/etcd:2.0.12 /usr/local/bin/etcd --addr=127.0.0.1:4001 --bind-addr=0.0.0.0:4001 --data-dir=/var/etcd/data
{% endraw %}
{% endhighlight %}

Next, you need to set a CIDR range for flannel.  This CIDR should be chosen to be non-overlapping with any existing network you are using:

{% highlight sh %}
{% raw %}
sudo docker -H unix:///var/run/docker-bootstrap.sock run --net=host gcr.io/google_containers/etcd:2.0.12 etcdctl set /coreos.com/network/config '{ "Network": "10.1.0.0/16" }'
{% endraw %}
{% endhighlight %}


### Set up Flannel on the master node

Flannel is a network abstraction layer build by CoreOS, we will use it to provide simplified networking between our Pods of containers.

Flannel re-configures the bridge that Docker uses for networking.  As a result we need to stop Docker, reconfigure its networking, and then restart Docker.

#### Bring down Docker

To re-configure Docker to use flannel, we need to take docker down, run flannel and then restart Docker.

Turning down Docker is system dependent, it may be:

{% highlight sh %}
{% raw %}
sudo /etc/init.d/docker stop
{% endraw %}
{% endhighlight %}

or

{% highlight sh %}
{% raw %}
sudo systemctl stop docker
{% endraw %}
{% endhighlight %}

or it may be something else.

#### Run flannel

Now run flanneld itself:

{% highlight sh %}
{% raw %}
sudo docker -H unix:///var/run/docker-bootstrap.sock run -d --net=host --privileged -v /dev/net:/dev/net quay.io/coreos/flannel:0.5.0
{% endraw %}
{% endhighlight %}

The previous command should have printed a really long hash, copy this hash.

Now get the subnet settings from flannel:

{% highlight sh %}
{% raw %}
sudo docker -H unix:///var/run/docker-bootstrap.sock exec <really-long-hash-from-above-here> cat /run/flannel/subnet.env
{% endraw %}
{% endhighlight %}

#### Edit the docker configuration

You now need to edit the docker configuration to activate new flags.  Again, this is system specific.

This may be in `/etc/default/docker` or `/etc/systemd/service/docker.service` or it may be elsewhere.

Regardless, you need to add the following to the docker command line:

{% highlight sh %}
{% raw %}
--bip=${FLANNEL_SUBNET} --mtu=${FLANNEL_MTU}
{% endraw %}
{% endhighlight %}

#### Remove the existing Docker bridge

Docker creates a bridge named `docker0` by default.  You need to remove this:

{% highlight sh %}
{% raw %}
sudo /sbin/ifconfig docker0 down
sudo brctl delbr docker0
{% endraw %}
{% endhighlight %}

You may need to install the `bridge-utils` package for the `brctl` binary.

#### Restart Docker

Again this is system dependent, it may be:

{% highlight sh %}
{% raw %}
sudo /etc/init.d/docker start
{% endraw %}
{% endhighlight %}

it may be:

{% highlight sh %}
{% raw %}
systemctl start docker
{% endraw %}
{% endhighlight %}

## Starting the Kubernetes Master

Ok, now that your networking is set up, you can startup Kubernetes, this is the same as the single-node case, we will use the "main" instance of the Docker daemon for the Kubernetes components.

{% highlight sh %}
{% raw %}
sudo docker run --net=host -d -v /var/run/docker.sock:/var/run/docker.sock  gcr.io/google_containers/hyperkube:v0.21.2 /hyperkube kubelet --api_servers=http://localhost:8080 --v=2 --address=0.0.0.0 --enable_server --hostname_override=127.0.0.1 --config=/etc/kubernetes/manifests-multi
{% endraw %}
{% endhighlight %}

### Also run the service proxy

{% highlight sh %}
{% raw %}
sudo docker run -d --net=host --privileged gcr.io/google_containers/hyperkube:v0.21.2 /hyperkube proxy --master=http://127.0.0.1:8080 --v=2
{% endraw %}
{% endhighlight %}

### Test it out

At this point, you should have a functioning 1-node cluster.  Let's test it out!

Download the kubectl binary
([OS X](http://storage.googleapis.com/kubernetes-release/release/v0.21.2/bin/darwin/amd64/kubectl))
([linux](http://storage.googleapis.com/kubernetes-release/release/v0.21.2/bin/linux/amd64/kubectl))

List the nodes

{% highlight sh %}
{% raw %}
kubectl get nodes
{% endraw %}
{% endhighlight %}

This should print:

{% highlight console %}
{% raw %}
NAME        LABELS                             STATUS
127.0.0.1   kubernetes.io/hostname=127.0.0.1   Ready
{% endraw %}
{% endhighlight %}

If the status of the node is `NotReady` or `Unknown` please check that all of the containers you created are successfully running.
If all else fails, ask questions on IRC at [#google-containers](http://webchat.freenode.net/?channels=google-containers).


### Next steps

Move on to [adding one or more workers](worker.html)


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker-multinode/master.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

