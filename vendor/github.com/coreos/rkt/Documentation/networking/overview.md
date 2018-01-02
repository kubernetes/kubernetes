# Networking

On some of rkt's subcommands *([run][rkt-run], [run-prepared][rkt-run-prepared])*, the `--net` flag allows you to configure the pod's network.
The various options can be grouped by two categories:

* [host mode](#host-mode)
* [contained mode (default)](#contained-mode)

This document gives a brief overview of the supported plugins.
More examples and advanced topics are linked in the [more docs](#more-docs) section.

## Host mode

When `--net=host` is passed the pod's apps will inherit the network namespace of the process that is invoking rkt.

If rkt is directly called from the host the apps within the pod will share the network stack and the interfaces with the host machine.
This means that every network service that runs in the pod has the same connectivity as if it was started on the host directly.

Applications that run in a pod which shares the host network namespace are able to access everything associated with the host's network interfaces: IP addresses, routes, iptables rules and sockets, including abstract Linux sockets.
Depending on the host's setup these abstract Linux sockets, used by applications like X11 and D-Bus, might expose critical endpoints to the pod's applications.
This risk can be avoided by configuring a separate namespace for pod.

## Contained mode

If anything other than `host` is passed to `--net=`, the pod will live in a separate network namespace with the help of [CNI][cni] and its plugin system.
The network setup for the pod's network namespace depends on the available CNI configuration files that are shipped with rkt and also configured by the user.

### Network selection

Every network must have a unique name and can only be joined once by every pod.
Passing a list of comma separated network as in `--net=net1,net2,net3,...` tells rkt which networks should be joined.
This is useful for grouping certain pod networks together while separating others.
There is also the possibility to load all configured networks by using  `--net=all`.

### Builtin networks

rkt ships with two built-in networks, named *default* and *default-restricted*.

### The default network

The *default* network is loaded automatically in three cases:

* `--net` is not present on the command line
* `--net` is passed with no options
* `--net=default`is passed

It consists of a loopback device and a veth device.
The veth pair creates a point-to-point link between the pod and the host.
rkt will allocate an IPv4 address out of 172.16.28.0/24 for the pod's veth interface.
It will additionally set the default route in the pod namespace.
Finally, it will enable IP masquerading on the host to NAT the egress traffic.

**Note**: The default network must be explicitly listed in order to be loaded when `--net=n1,n2,...` is specified with a list of network names.

Example: If you want default networking and two more networks you need to pass `--net=default,net1,net2`.

### The default-restricted network

The *default-restricted* network does not set up the default route and IP masquerading.
It only allows communication with the host via the veth interface and thus enables the pod to communicate with the metadata service which runs on the host.
If *default* is not among the specified networks, the *default-restricted* network will be added to the list of networks automatically.
It can also be loaded directly by explicitly passing `--net=default-restricted`.

### No (loopback only) networking

The passing of `--net=none` will put the pod in a network namespace with only the loopback networking.
This can be used to completely isolate the pod's network.

```sh
$ sudo rkt run --interactive --net=none kinvolk.io/aci/busybox:1.24
(...)
/ # ip address
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue
	link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
	inet 127.0.0.1/8 scope host lo
   	valid_lft forever preferred_lft forever
	inet6 ::1/128 scope host
   	valid_lft forever preferred_lft forever
/ # ip route
/ # ping localhost
PING localhost (127.0.0.1): 56 data bytes
64 bytes from 127.0.0.1: seq=0 ttl=64 time=0.022 ms
^C
```

The situation here is very straightforward: no routes, the interface _lo_ with the local address.
The resolution of localhost is enabled in rkt by default, as it will generate a minimal `/etc/hosts` inside the pod if the image does not provide one.

### Setting up additional networks

In addition to the default network (veth) described in the previous sections, rkt pods can be configured to join additional networks.
Each additional network will result in an new interface being set up in the pod.
The type of network interface, IP, routes, etc is controlled via a configuration file residing in `/etc/rkt/net.d` directory.
The network configuration files are executed in lexicographically sorted order.
Each file consists of a JSON dictionary as shown below:

```json
$ cat /etc/rkt/net.d/10-containers.conf
{
	"name": "containers",
	"type": "bridge",
	"ipam": {
		"type": "host-local",
		"subnet": "10.1.0.0/16"
	}
}
```

This configuration file defines a linux-bridge based network on 10.1.0.0/16 subnet.
The following fields apply to all configuration files.
Additional fields are specified for various types.

- **name** (string): an arbitrary label for the network.
  By convention the conf file is named with a leading ordinal, dash, network name, and .conf extension.
- **type** (string): the type of network/interface to create.
  The type actually names a network plugin.
  rkt is bundled with some built-in plugins.
- **ipam** (dict): IP Address Management -- controls the settings related to IP address assignment, gateway, and routes.

### Built-in network types

#### ptp

ptp is probably the simplest type of networking and is used to set up default network.
It creates a virtual ethernet pair (akin to a pipe) and places one end into pod and the other on the host.

`ptp` specific configuration fields are:

- **mtu** (integer): the size of the MTU in bytes.
- **ipMasq** (boolean): whether to set up IP masquerading on the host.

#### bridge

Like the ptp type, `bridge` will create a veth pair and attach one end to the pod.
However the host end of the veth will be plugged into a linux-bridge.
The configuration file specifies the bridge name and if the bridge does not exist, it will be created.
The bridge can optionally be configured to act as the gateway for the network.

`bridge` specific configuration fields are:

- **bridge** (string): the name of the bridge to create and/or plug into.
  Defaults to `rkt0`.
- **isGateway** (boolean): whether the bridge should be assigned an IP and act as a gateway.
- **mtu** (integer): the size of the MTU in bytes for bridge and veths.
- **ipMasq** (boolean): whether to set up IP masquerading on the host.

#### macvlan

macvlan behaves similar to a bridge but does not provide communication between the host and the pod.

macvlan creates a virtual copy of a master interface and assigns the copy a randomly generated MAC address.
The pod can communicate with the network that is attached to the master interface.
The distinct MAC address allows the pod to be identified by external network services like DHCP servers, firewalls, routers, etc.
macvlan interfaces cannot communicate with the host via the macvlan interface.
This is because traffic that is sent by the pod onto the macvlan interface is bypassing the master interface and is sent directly to the interfaces underlying network.
Before traffic gets sent to the underlying network it can be evaluated within the macvlan driver, allowing it to communicate with all other pods that created their macvlan interface from the same master interface.

`macvlan` specific configuration fields are:

- **master** (string): the name of the host interface to copy.
  This field is required.
- **mode** (string): one of "bridge", "private", "vepa", or "passthru".
  This controls how traffic is handled between different macvlan interfaces on the same host.
  See [this guide][macvlan-modes] for discussion of modes.
  Defaults to "bridge".
- **mtu** (integer): the size of the MTU in bytes for the macvlan interface.
  Defaults to MTU of the master device.
- **ipMasq** (boolean): whether to set up IP masquerading on the host.
  Defaults to false.

#### ipvlan

ipvlan behaves very similar to macvlan but does not provide distinct MAC addresses for pods.
macvlan and ipvlan can't be used on the same master device together.

ipvlan creates virtual copies of interfaces like macvlan but does not assign a new MAC address to the copied interface.
This does not allow the pods to be distinguished on a MAC level and so cannot be used with DHCP servers.
In other scenarios this can be an advantage, e.g. when an external network port does not allow multiple MAC addresses.
ipvlan also solves the problem of MAC address exhaustion that can occur with a large number of pods copying the same master interface.
ipvlan interfaces are able to have different IP addresses than the master interface and will therefore have the needed distinction for most use-cases.

`ipvlan` specific configuration fields are:
- **master** (string): the name of the host interface to copy.
  This field is required.
- **mode** (string): one of "l2", "l3".
  See [kernel documentation on ipvlan][ipvlan].
  Defaults to "l2".
- **mtu** (integer): the size of the MTU in bytes for the ipvlan interface.
  Defaults to MTU of the master device.
- **ipMasq** (boolean): whether to set up IP masquerading on the host.
  Defaults to false.

**Notes**

* ipvlan can cause problems with duplicated IPv6 link-local addresses since they are partially constructed using the MAC address.
  This issue is being currently addressed by the ipvlan kernel module developers.

## IP Address Management

The policy for IP address allocation, associated gateway and routes is separately configurable via the `ipam` section of the configuration file.
rkt currently ships with two IPAM types: host-local and DHCP.
Like the network types, IPAM types can be implemented by third-parties via plugins.

### host-local

host-local type allocates IPs out of specified network range, much like a DHCP server would.
The difference is that while DHCP uses a central server, this type uses a static configuration.
Consider the following conf:

```json
$ cat /etc/rkt/net.d/10-containers.conf
{
	"name": "containers",
	"type": "bridge",
	"bridge": "rkt1",
	"ipam": {
		"type": "host-local",
		"subnet": "10.1.0.0/16"
	}
}
```

This configuration instructs rkt to create `rkt1` Linux bridge and plugs pods into it via veths.
Since the subnet is defined as `10.1.0.0/16`, rkt will assign individual IPs out of that range.
The first pod will be assigned 10.1.0.2/16, next one 10.1.0.3/16, etc (it reserves 10.1.0.1/16 for gateway).
Additional configuration fields:

- **subnet** (string): subnet in CIDR notation for the network.
- **rangeStart** (string): first IP address from which to start allocating IPs.
  Defaults to second IP in `subnet` range.
- **rangeEnd** (string): last IP address in the allocatable range.
  Defaults to last IP in `subnet` range.
- **gateway** (string): the IP address of the gateway in this subnet.
- **routes** (list of strings): list of IP routes in CIDR notation.
  The routes get added to pod namespace with next-hop set to the gateway of the network.

The following shows a more complex IPv6 example in combination with the ipvlan plugin.
The gateway is configured for the default route, allowing the pod to access external networks via the ipvlan interface.

```json
{
    "name": "ipv6-public",
    "type": "ipvlan",
    "master": "em1",
    "mode": "l3",
    "ipam": {
        "type": "host-local",
        "subnet": "2001:0db8:161:8374::/64",
        "rangeStart": "2001:0db8:161:8374::1:2",
        "rangeEnd": "2001:0db8:161:8374::1:fffe",
        "gateway": "fe80::1",
        "routes": [
            { "dst": "::0/0" }
        ]
    }
}
```

### dhcp

The DHCP type requires a special client daemon, part of the [CNI DHCP plugin][cni-dhcp], to be running on the host.
This acts as a proxy between a DHCP client running inside the container and a DHCP service already running on the network, as well as renewing leases appropriately.

The DHCP plugin binary can be executed in the daemon mode by launching it with `daemon` argument.
However, in rkt the DHCP plugin is bundled in stage1.aci so this requires extracting the binary from it:

```
$ sudo ./rkt fetch --insecure-options=image ./stage1.aci
$ sudo ./rkt image extract coreos.com/rkt/stage1 /tmp/stage1
$ sudo cp /tmp/stage1/rootfs/usr/lib/rkt/plugins/net/dhcp .
```

Now start the daemon:

```
$ sudo ./dhcp daemon
```

It is now possible to use the DHCP type by specifying it in the ipam section of the network configuration file:

```json
{
	"name": "lan",
	"type": "macvlan",
	"master": "eth0",
	"ipam": {
		"type": "dhcp"
	}
}
```

For more information about the DHCP plugin, see the [CNI docs][cni-dhcp].

## Other plugins

### flannel

This plugin is designed to work in conjunction with flannel, a network fabric for containers.
The basic network configuration is as follows:

```json
{
	"name": "containers",
	"type": "flannel"
}
```

This will set up a linux-bridge, connect the container to the bridge and assign container IPs out of the subnet that flannel assigned to the host.
For more information included advanced configuration options, see [CNI docs][cni-flannel].

## Exposing container ports on the host

Apps declare their public ports in the image manifest file.
A user can expose some or all of these ports to the host when running a pod.
Doing so allows services inside the pods to be reachable through the host's IP address.

The example below demonstrates an image manifest snippet declaring a single port:

```json
"ports": [
	{
		"name": "http",
		"port": 80,
		"protocol": "tcp"
	}
]
```

The pod's TCP port 80 can be mapped to an arbitrary port on the host during rkt invocation:

```
# rkt run --port=http:8888 myapp.aci
```

Now, any traffic arriving on host's TCP port 8888 will be forwarded to the pod on port 80.

### Network used for forwarded ports

The network that will be chosen for the port forwarding depends on the _ipMasq_ setting of the configured networks.
If at least one of them has _ipMasq_ enabled, the forwarded traffic will be passed through the first loaded network that has IP masquerading enabled.
If no network is masqueraded, the last loaded network will be used.
As a reminder, the sort order of the loaded networks is detailed in the chapter about [setting up additional networks](#setting-up-additional-networks).

### Socket Activation
rkt also supports socket activation.
This is documented in [Socket-activated service][socket-activated].

## More Docs

##### Examples
* [bridge plugin][examples-bridge]

##### Other topics:
* [DNS configuration][dns]
* [Overriding defaults][overriding]


[cni]: https://github.com/appc/cni
[cni-dhcp]: https://github.com/appc/cni/blob/master/Documentation/dhcp.md
[cni-flannel]: https://github.com/appc/cni/blob/master/Documentation/flannel.md
[dns]: dns.md
[examples-bridge]: examples-bridge.md
[ipvlan]: https://www.kernel.org/doc/Documentation/networking/ipvlan.txt
[macvlan-modes]: http://www.pocketnix.org/posts/Linux%20Networking:%20MAC%20VLANs%20and%20Virtual%20Ethernets
[overriding]: overriding-defaults.md
[rkt-run]: ../subcommands/run.md
[rkt-run-prepared]: ../subcommands/run-prepared.md
[socket-activated]: ../using-rkt-with-systemd.md#socket-activated-service
