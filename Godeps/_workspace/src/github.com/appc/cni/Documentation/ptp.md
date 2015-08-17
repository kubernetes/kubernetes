# ptp plugin

## Overview
The ptp plugin creates a point-to-point link between a container and the host by using a veth device.
One end of the veth pair is placed inside a container and the other end resides on the host.
Both ends receive an IP address out of a /31 range.
The IP of the host end becomes the gateway address inside the container.

Because ptp plugin requires a pair of IP addresses for each container, it should be used in conjuction with host-local-ptp IPAM plugin.

## Example network configuration
```
{
	"name": "mynet",
	"type": "ptp",
	"ipam": {
		"type": "host-local-ptp",
		"subnet": "10.1.1.0/24"
	}
}

## Network configuration reference

* `name` (string, required): the name of the network
* `type` (string, required): "ptp"
* `ipMasq` (boolean, optional): set up IP Masquerade on the host for traffic originating from this network and destined outside of it. Defaults to false.
* `mtu` (integer, optional): explicitly set MTU to the specified value. Defaults to value chosen by the kernel.
* `ipam` (dictionary, required): IPAM configuration to be used for this network.
