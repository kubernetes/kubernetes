<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/proposals/flannel-integration.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Flannel integration with Kubernetes

## Why?

* Networking works out of the box.
* Cloud gateway configuration is regulated by quota.
* Consistent bare metal and cloud experience.
* Lays foundation for integrating with networking backends and vendors.

## How?

Thus:

```
Master                      |               Node1
----------------------------------------------------------------------
{192.168.0.0/16, 256 /24}   |               docker
    |                       |                 | restart with podcidr
apiserver            <------------------    kubelet (sends podcidr)
    |                       |                 | here's podcidr, mtu
flannel-server:10253 <------------------    flannel-daemon
Allocates a /24      ------------------>    [config iptables, VXLan]
                     <------------------    [watch subnet leases]
I just allocated     ------------------>    [config VXLan]
another /24                 |
```

## Proposal

Explaining vxlan is out of the scope of this document, however it does take some basic understanding to grok the proposal. Assume some pod wants to communicate across nodes with the above setup. Check the flannel vxlan devices:

```console
node1 $ ip -d link show flannel.1
4: flannel.1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1410 qdisc noqueue state UNKNOWN mode DEFAULT
    link/ether a2:53:86:b5:5f:c1 brd ff:ff:ff:ff:ff:ff
    vxlan
node1 $ ip -d link show eth0
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1460 qdisc mq state UP mode DEFAULT qlen 1000
    link/ether 42:01:0a:f0:00:04 brd ff:ff:ff:ff:ff:ff

node2 $ ip -d link show flannel.1
4: flannel.1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1410 qdisc noqueue state UNKNOWN mode DEFAULT
    link/ether 56:71:35:66:4a:d8 brd ff:ff:ff:ff:ff:ff
    vxlan
node2 $ ip -d link show eth0
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1460 qdisc mq state UP mode DEFAULT qlen 1000
    link/ether 42:01:0a:f0:00:03 brd ff:ff:ff:ff:ff:ff
```

Note that we're ignoring cbr0 for the sake of simplicity. Spin-up a container on each node. We're using raw docker for this example only because we want control over where the container lands:

```
node1 $ docker run -it radial/busyboxplus:curl /bin/sh
[ root@5ca3c154cde3:/ ]$ ip addr show
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue
8: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1410 qdisc noqueue
    link/ether 02:42:12:10:20:03 brd ff:ff:ff:ff:ff:ff
    inet 192.168.32.3/24 scope global eth0
       valid_lft forever preferred_lft forever

node2 $ docker run -it radial/busyboxplus:curl /bin/sh
[ root@d8a879a29f5d:/ ]$ ip addr show
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue
16: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1410 qdisc noqueue
    link/ether 02:42:12:10:0e:07 brd ff:ff:ff:ff:ff:ff
    inet 192.168.14.7/24 scope global eth0
       valid_lft forever preferred_lft forever
[ root@d8a879a29f5d:/ ]$ ping 192.168.32.3
PING 192.168.32.3 (192.168.32.3): 56 data bytes
64 bytes from 192.168.32.3: seq=0 ttl=62 time=1.190 ms
```

__What happened?__:

From 1000 feet:
* vxlan device driver starts up on node1 and creates a udp tunnel endpoint on 8472
* container 192.168.32.3 pings 192.168.14.7
    - what's the MAC of 192.168.14.0?
        - L2 miss, flannel looks up MAC of subnet
        - Stores `192.168.14.0 <-> 56:71:35:66:4a:d8` in neighbor table
    - what's tunnel endpoint of this MAC?
        - L3 miss, flannel looks up destination VM ip
        - Stores `10.240.0.3 <-> 56:71:35:66:4a:d8` in bridge database
* Sends `[56:71:35:66:4a:d8, 10.240.0.3][vxlan: port, vni][02:42:12:10:20:03, 192.168.14.7][icmp]`

__But will it blend?__

Kubernetes integration is fairly straight-forward once we understand the pieces involved, and can be prioritized as follows:
* Kubelet understands flannel daemon in client mode, flannel server manages independent etcd store on master, node controller backs off CIDR allocation
* Flannel server consults the Kubernetes master for everything network related
* Flannel daemon works through network plugins in a generic way without bothering the kubelet: needs CNI x Kubernetes standardization

The first is accomplished in this PR, while a timeline for 2. and 3. is TDB. To implement the flannel api we can either run a proxy per node and get rid of the flannel server, or service all requests in the flannel server with something like a go-routine per node:
* `/network/config`: read network configuration and return
* `/network/leases`:
	- Post:  Return a lease as understood by flannel
		- Lookip node by IP
		- Store node metadata from [flannel request] (https://github.com/coreos/flannel/blob/master/subnet/subnet.go#L34) in annotations
		- Return [Lease object] (https://github.com/coreos/flannel/blob/master/subnet/subnet.go#L40) reflecting node cidr
	- Get: Handle a watch on leases
* `/network/leases/subnet`:
	- Put: This is a request for a lease. If the nodecontroller is allocating CIDRs we can probably just no-op.
* `/network/reservations`: TDB, we can probably use this to accommodate node controller allocating CIDR instead of flannel requesting it

The ick-iest part of this implementation is going to the `GET /network/leases`, i.e the watch proxy. We can side-step by waiting for a more generic Kubernetes resource. However, we can also implement it as follows:
* Watch all nodes, ignore heartbeats
* On each change, figure out the lease for the node, construct a [lease watch result](https://github.com/coreos/flannel/blob/0bf263826eab1707be5262703a8092c7d15e0be4/subnet/subnet.go#L72), and send it down the watch with the RV from the node
* Implement a lease list that does a similar translation

I say this is gross without an api object because for each node->lease translation one has to store and retrieve the node metadata sent by flannel (eg: VTEP) from node annotations. [Reference implementation](https://github.com/bprashanth/kubernetes/blob/network_vxlan/pkg/kubelet/flannel_server.go) and [watch proxy](https://github.com/bprashanth/kubernetes/blob/network_vxlan/pkg/kubelet/watch_proxy.go).

# Limitations

* Integration is experimental
* Flannel etcd not stored in persistent disk
* CIDR allocation does *not* flow from Kubernetes down to nodes anymore

# Wishlist

This proposal is really just a call for community help in writing a Kubernetes x flannel backend.

* CNI plugin integration
* Flannel daemon in privileged pod
* Flannel server talks to apiserver, described in proposal above
* HTTPs between flannel daemon/server
* Investigate flannel server running on every node (as done in the reference implementation mentioned above)
* Use flannel reservation mode to support node controller podcidr allocation


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/flannel-integration.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
