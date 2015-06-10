# Networking in Kubernetes

## Summary

Kubernetes approaches networking somewhat differently that Docker's defaults.
We give every pod its own IP address allocated from an internal network, so you
do not need to explicitly create links between communicating pods.  To do this,
you must set up your cluster networking correctly.

Since pods can fail and be replaced with new pods with different IP addresses
on different nodes, we do not recommend having a pod directly talk to the IP
address of another Pod.  Instead, if a pod, or collection of pods, provide some
service, then you should create a `service` object spanning those pods, and
clients should connect to the IP of the service object.  See
[services](services.md).

## Docker model

Before discussing the Kubernetes approach to networking, it is worthwhile to
review the "normal" way that networking works with Docker.  By default, Docker
uses host-private networking.  It creates a virtual bridge, called `docker0` by
default, and allocates a subnet from one of the private address blocks defined
in [RFC1918](https://tools.ietf.org/html/rfc1918) for that bridge.  For each
container that Docker creates, it allocates a virtual ethernet device (called
`veth`) which is attached to the bridge. The veth is mapped to appear as eth0
in the container, using Linux namespaces.  The in-container eth0 interface is
given an IP address from the bridge's address range.

The result is that Docker containers can talk to other containers only if they
are on the same machine (and thus the same virtual bridge).  Containers on
different machines can not reach each other - in fact they may end up with the
exact same network ranges and IP addresses.

In order for Docker containers to communicate across nodes, they must be
allocated ports on the machine's own IP address, which are then forwarded or
proxied to the containers.  This obviously means that containers must either
coordinate which ports they use very carefully or else be allocated ports
dynamically.

## Kubernetes model

Coordinating ports across multiple developers is very difficult to do at
scale and exposes users to cluster-level issues outside of their control.
Dynamic port allocation brings a lot of complications to the system - every
application has to take ports as flags, the API servers have to know how to
insert dynamic port numbers into configuration blocks, services have to know
how to find each other, etc.  Rather than deal with this, Kubernetes takes a
different approach.

Kubernetes imposes the following fundamental requirements on any networking
implementation (barring any intentional network segmentation policies):
   * all containers can communicate with all other containers without NAT
   * all nodes can communicate with all containers (and vice-versa) without NAT
   * the IP that a container sees itself as is the same IP that others see it as

What this means in practice is that you can not just take two computers
running Docker and expect Kubernetes to work.  You must ensure that the
fundamental requirements are met.

This model is not only less complex overall, but it is principally compatible
with the desire for Kubernetes to enable low-friction porting of apps from VMs
to containers.  If your job previously ran in a VM, your VM had an IP and could
talk to other VMs in your project.  This is the same basic model.

Until now this document has talked about containers.  In reality, Kubernetes
applies IP addresses at the `Pod` scope - containers within a `Pod` share their
network namespaces - including their IP address.  This means that containers
within a `Pod` can all reach each other’s ports on `localhost`.  This does imply
that containers within a `Pod` must coordinate port usage, but this is no
different that processes in a VM.  We call this the "IP-per-pod" model.  This
is implemented in Docker as a "pod container" which holds the network namespace
open while "app containers" (the things the user specified) join that namespace
with Docker's `--net=container:<id>` function.

As with Docker, it is possible to request host ports, but this is reduced to a
very niche operation.  In this case a port will be allocated on the host `Node`
and traffic will be forwarded to the `Pod`.  The `Pod` itself is blind to the
existence or non-existence of host ports.

## How to achieve this

There are a number of ways that this network model can be implemented.  This
document is not an exhaustive study of the various methods, but hopefully serves
as an introduction to various technologies and serves as a jumping-off point.
If some techniques become vastly preferable to others, we might detail them more
here.

### Google Compute Engine

For the Google Compute Engine cluster configuration scripts, we use [advanced
routing](https://developers.google.com/compute/docs/networking#routing) to
assign each VM a subnet (default is /24 - 254 IPs).  Any traffic bound for that
subnet will be routed directly to the VM by the GCE network fabric.  This is in
addition to the "main" IP address assigned to the VM, which is NAT'ed for
outbound internet access.  A linux bridge (called `cbr0`) is configured to exist
on that subnet, and is passed to docker's `--bridge` flag.

We start Docker with:

```
    DOCKER_OPTS="--bridge cbr0 --iptables=false --ip-masq=false"
```

We set up this bridge on each node with SaltStack, in
[container_bridge.py](../cluster/saltbase/salt/_states/container_bridge.py).

```
cbr0:
  container_bridge.ensure:
    - cidr: {{ grains['cbr-cidr'] }}
    - mtu: 1460
```

Docker will now allocate `Pod` IPs from the `cbr-cidr` block.  Containers
can reach each other and `Nodes` over the `cbr0` bridge.  Those IPs are all
routable within the GCE project network.

GCE itself does not know anything about these IPs, though, so it will not NAT
them for outbound internet traffic.  To achieve that we use an iptables rule to
masquerade (aka SNAT - to make it seem as if packets came from the `Node`
itself) traffic that is bound for IPs outside the GCE project network
(10.0.0.0/8).

```
iptables -t nat -A POSTROUTING ! -d 10.0.0.0/8 -o eth0 -j MASQUERADE
```

Lastly we enable IP forwarding in the kernel (so the kernel will process
packets for bridged containers):

```
sysctl net.ipv4.ip_forward=1
```

The result of all this is that all `Pods` can reach each other and can egress
traffic to the internet.

### L2 networks and linux bridging

If you have a "dumb" L2 network, such as a simple switch in a "bare-metal"
environment, you should be able to do something similar to the above GCE setup.
Note that these instructions have only been tried very casually - it seems to
work, but has not been thoroughly tested.  If you use this technique and
perfect the process, please let us know.

Follow the "With Linux Bridge devices" section of [this very nice
tutorial](http://blog.oddbit.com/2014/08/11/four-ways-to-connect-a-docker/) from
Lars Kellogg-Stedman.

### Flannel

[Flannel](https://github.com/coreos/flannel#flannel) is a very simple overlay
network that satisfies the Kubernetes requirements.  It installs in minutes and
should get you up and running if the above techniques are not working.  Many
people have reported success with Flannel and Kubernetes.

### OpenVSwitch

[OpenVSwitch](./ovs-networking.md) is a somewhat more mature but also
complicated way to build an overlay network.  This is endorsed by several of the
"Big Shops" for networking.

### Weave

[Weave](https://github.com/zettio/weave) is yet another way to build an overlay
network, primarily aiming at Docker integration.

### Calico

[Calico](https://github.com/Metaswitch/calico) uses BGP to enable real container
IPs.

## Other reading

The early design of the networking model and its rationale, and some future
plans are described in more detail in the [networking design
document](design/networking.md).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/networking.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/networking.md?pixel)]()
