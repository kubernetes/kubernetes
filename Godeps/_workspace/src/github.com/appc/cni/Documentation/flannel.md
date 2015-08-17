# flannel plugin

## Overview
This plugin is designed to work in conjunction with [flannel](https://github.com/coreos/flannel), a network fabric for containers.
When flannel daemon is started, it outputs a `/run/flannel/subnet.env` file that looks like this:
```
FLANNEL_SUBNET=10.1.17.0/24
FLANNEL_MTU=1472
FLANNEL_IPMASQ=true
```

This information reflects the attributes of flannel network on the host.
The flannel CNI plugin uses this information to configure another CNI plugin, such as bridge plugin.

## Operation
Given the following network configuration file and the contents of `/run/flannel/subnet.env` above,
```
{
	"name": "mynet",
	"type": "flannel"
}
```
the flannel plugin will generate another network configuration file:
```
{
	"name": "mynet",
	"type": "bridge",
	"mtu": 1472,
	"ipMasq": false,
	"isGateway": true,
	"ipam": {
		"type": "host-local",
		"subnet": "10.1.17.0/24"
	}
}
```

It will then invoke the bridge plugin, passing it the generated configuration.

As can be seen from above, the flannel plugin, by default, will delegate to the bridge plugin.
If additional configuration values need to be passed to the bridge plugin, it can be done so via the `delegate` field:
```
{
	"name": "mynet",
	"type": "flannel",
	"delegate": {
		"bridge": "mynet0",
		"mtu": 1400
	}
}
```

This supplies a configuration parameter to the bridge plugin -- the created bridge will now be named `mynet0`.
Notice that `mtu` has also been specified and this value will not be overwritten by flannel plugin.

Additionally, the `delegate` field can be used to select a different kind of plugin altogether.
To use `ipvlan` instead of `bridge`, the following configuratoin can be specified:

```
{
	"name": "mynet",
	"type": "flannel",
	"delegate": {
		"type": "ipvlan",
		"master": "eth0"
	}
}
```

## Network configuration reference

* `name` (string, required): the name of the network
* `type` (string, required): "flannel"
* `subnetFile` (string, optional): full path to the subnet file written out by flanneld. Defaults to /run/flannel/subnet.env
* `delegate` (dictionary, optional): specifies configuration options for the delegated plugin.

flannel plugin will always set the following fields in the delegated plugin configuration:

* `name`: value of its "name" field.
* `ipam`: "host-local" type will be used with "subnet" set to `$FLANNEL_SUBNET`.

flannel plugin will set the following fields in the delegated plugin configuration if they are not present:
* `ipMasq`: the inverse of `$FLANNEL_IPMASQ`
* `mtu`: `$FLANNEL_MTU`

Additionally, for the bridge plugin, `isGateway` will be set to `true`, if not present.
