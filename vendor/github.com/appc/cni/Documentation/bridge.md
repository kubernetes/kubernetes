# bridge plugin

## Overview

With bridge plugin, all containers (on the same host) are plugged into a bridge (virtual switch) that resides in the host network namespace.
The containers receive one end of the veth pair with the other end connected to the bridge.
An IP address is only assigned to one end of the veth pair -- one residing in the container.
The bridge itself can also be assigned an IP address, turning it into a gateway for the containers.
Alternatively, the bridge can function purely in L2 mode and would need to be bridged to the host network interface (if other than container-to-container communication on the same host is desired).

The network configuration specifies the name of the bridge to be used.
If the bridge is missing, the plugin will create one on first use and, if gateway mode is used, assign it an IP that was returned by IPAM plugin via the gateway field.

## Example configuration
```
{
	"name": "mynet",
	"type": "bridge",
	"bridge": "mynet0",
	"isGateway": true,
	"ipMasq": true,
	"ipam": {
		"type": "host-local",
		"subnet": "10.10.0.0/16"
	}
}
```

## Network configuration reference

* `name` (string, required): the name of the network.
* `type` (string, required): "bridge".
* `bridge` (string, optional): name of the bridge to use/create. Defaults to "cni0".
* `isGateway` (boolean, optional): assign an IP address to the bridge. Defaults to false.
* `ipMasq` (boolean, optional): set up IP Masquerade on the host for traffic originating from this network and destined outside of it. Defaults to false.
* `mtu` (integer, optional): explicitly set MTU to the specified value. Defaults to the value chosen by the kernel.
* `ipam` (dictionary, required): IPAM configuration to be used for this network.
