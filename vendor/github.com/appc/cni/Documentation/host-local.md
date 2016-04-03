# host-local plugin

## Overview

host-local IPAM plugin allocates IPv4 addresses out of a specified address range.
It stores the state locally on the host filesystem, therefore ensuring uniqueness of IP addresses on a single host.

## Example configuration
```
{
	"ipam": {
		"type": "host-local",
		"subnet": "10.10.0.0/16",
		"rangeStart": "10.10.1.20",
		"rangeEnd": "10.10.3.50",
		"gateway": "10.10.0.254",
		"routes": [
			{ "dst": "0.0.0.0/0" },
			{ "dst": "192.168.0.0/16", "gw": "10.10.5.1" }
		]
	}
}
```

## Network configuration reference

* `type` (string, required): "host-local".
* `subnet` (string, required): CIDR block to allocate out of.
* `rangeStart` (string, optional): IP inside of "subnet" from which to start allocating addresses. Defaults to ".2" IP inside of the "subnet" block.
* `rangeEnd` (string, optional): IP inside of "subnet" with which to end allocating addresses. Defaults to ".254" IP inside of the "subnet" block.
* `gateway` (string, optional): IP inside of "subnet" to designate as the gateway. Defaults to ".1" IP inside of the "subnet" block.
* `routes` (string, optional): list of routes to add to the container namespace. Each route is a dictionary with "dst" and optional "gw" fields. If "gw" is omitted, value of "gateway" will be used.

## Supported arguments
The following [CNI_ARGS](https://github.com/appc/cni/blob/master/SPEC.md#parameters) are supported:

* `ip`: request a specific IP address from the subnet. If it's not available, the plugin will exit with an error

## Files

Allocated IP addresses are stored as files in /var/lib/cni/networks/$NETWORK_NAME.
