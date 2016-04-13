# ipvlan plugin

## Overview

ipvlan is a new [addition](https://lwn.net/Articles/620087/) to the Linux kernel.
Like its cousin macvlan, it virtualizes the host interface.
However unlike macvlan which generates a new MAC address for each interface, ipvlan devices all share the same MAC.
The kernel driver inspects the IP address of each packet when making a decision about which virtual interface should process the packet.

Because all ipvlan interfaces share the MAC address with the host interface, DHCP can only be used in conjunction with ClientID (currently not supported by DHCP plugin).

## Example configuration

```
{
	"name": "mynet",
	"type": "ipvlan",
	"master": "eth0",
	"ipam": {
		"type": "host-local",
		"subnet": "10.1.2.0/24",
	}
}
```

## Network configuration reference

* `name` (string, required): the name of the network.
* `type` (string, required): "ipvlan".
* `master` (string, required): name of the host interface to enslave.
* `mode` (string, optional): one of "l2", "l3". Defaults to "l2".
* `mtu` (integer, optional): explicitly set MTU to the specified value. Defaults to the value chosen by the kernel.
* `ipam` (dictionary, required): IPAM configuration to be used for this network.

## Notes

* `ipvlan` does not allow virtual interfaces to communicate with the master interface.
Therefore the container will not be able to reach the host via `ipvlan` interface.
Be sure to also have container join a network that provides connectivity to the host (e.g. `ptp`).
* A single master interface can not be enslaved by both `macvlan` and `ipvlan`.
