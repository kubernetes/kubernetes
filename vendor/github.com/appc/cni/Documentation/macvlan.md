# macvlan plugin

## Overview

[macvlan](http://backreference.org/2014/03/20/some-notes-on-macvlanmacvtap/) functions like a switch that is already connected to the host interface.
A host interface gets "enslaved" with the virtual interfaces sharing the physical device but having distinct MAC addresses.
Since each macvlan interface has its own MAC address, it makes it easy to use with existing DHCP servers already present on the network.

## Example configuration

```
{
	"name": "mynet",
	"type": "macvlan",
	"master": "eth0",
	"ipam": {
		"type": "dhcp"
	}
}
```

## Network configuration reference

* `name` (string, required): the name of the network
* `type` (string, required): "macvlan"
* `master` (string, required): name of the host interface to enslave
* `mode` (string, optional): one of "bridge", "private", "vepa", "passthrough". Defaults to "bridge".
* `mtu` (integer, optional): explicitly set MTU to the specified value. Defaults to the value chosen by the kernel.
* `ipam` (dictionary, required): IPAM configuration to be used for this network.

## Notes

* If are testing on a laptop, please remember that most wireless cards do not support being enslaved by macvlan.
* A single master interface can not be enslaved by both `macvlan` and `ipvlan`.
