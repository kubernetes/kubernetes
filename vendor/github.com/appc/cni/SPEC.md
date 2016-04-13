# Container Networking Interface Proposal

## Overview

This document proposes a generic plugin-based networking solution for application containers on Linux, the _Container Networking Interface_, or _CNI_. It is derived from the [rkt Networking Proposal][rkt-networking-proposal], which aimed to satisfy many of the [design considerations][rkt-networking-design] for networking in [rkt][rkt-github].

For the purposes of this proposal, we define two terms very specifically:
- _container_ can be considered synonymous with a [Linux _network namespace_][namespaces]. What unit this corresponds to depends on a particular container runtime implementation: for example, in implementations of the [App Container Spec][appc-github] like rkt, each _pod_ runs in a unique network namespace. In [Docker][docker], on the other hand, network namespaces generally exist for each separate Docker container.
- _network_ refers to a group of entities that are uniquely addressable that can communicate amongst each other. This could be either an individual container (as specified above), a machine, or some other network device (e.g. a router). Containers can be conceptually _added to_ or _removed from_ one or more networks.

[rkt-networking-proposal]: https://docs.google.com/a/coreos.com/document/d/1PUeV68q9muEmkHmRuW10HQ6cHgd4819_67pIxDRVNlM/edit#heading=h.ievko3xsjwxd
[rkt-networking-design]: 
https://docs.google.com/a/coreos.com/document/d/1CTAL4gwqRofjxyp4tTkbgHtAwb2YCcP14UEbHNizd8g
[rkt-github]: https://github.com/coreos/rkt
[namespaces]: http://man7.org/linux/man-pages/man7/namespaces.7.html 
[appc-github]: https://github.com/appc/spec
[docker]: https://docker.com 

## General considerations

The intention is for the container runtime to first create a new network namespace for the container.
It then determines which networks this container should belong to and for each network, which plugin must be executed.
The network configuration is in JSON format and can easily be stored in a file.
The network configuration includes mandatory fields such as "name" and "type" as well as plugin (type) specific ones.
The container runtime sequentially sets up the networks by executing the corresponding plugin for each network.
Upon completion of the container lifecycle, the runtime executes the plugins in reverse order (relative to the order in which they were added) to disconnect them from the networks.

## CNI Plugin

### Overview

Each CNI plugin is implemented as an executable that is invoked by the container management system (e.g. rkt or Docker).

A CNI plugin is responsible for inserting a network interface into the container network namespace (e.g. one end of a veth pair) and making any necessary changes on the host (e.g. attaching other end of veth into a bridge).
It should then assign the IP to the interface and setup the routes consistent with IP Address Management section by invoking appropriate IPAM plugin.

### Parameters

The operations that the CNI plugin needs to support are:


- Add container to network
  - Parameters:
    - **Container ID**. This is optional but recommended, and should be unique across an administrative domain while the container is live (it may be reused in the future). For example, an environment with an IPAM system may require that each container is allocated a unique ID and that each IP allocation can thus be correlated back to a particular container. As another example, in appc implementations this would be the _pod ID_.
    - **Network namespace path**. This represents the path to the network namespace to be added, i.e. /proc/[pid]/ns/net or a bind-mount/link to it.
    - **Network configuration**. This is a JSON document describing a network to which a container can be joined. The schema is described below.
    - **Extra arguments**. This allows granular configuration of CNI plugins on a per-container basis.
    - **Name of the interface inside the container**. This is the name that should be assigned to the interface created inside the container (network namespace); consequently it must comply with the standard Linux restrictions on interface names.
  - Result:
    - **IPs assigned to the interface**. This is either an IPv4 address, an IPv6 address, or both.

- Delete container from network
  - Parameters:
    - **Container ID**, as defined above.
    - **Network namespace path**, as defined above.
    - **Network configuration**, as defined above.
    - **Extra arguments**, as defined above.
    - **Name of the interface inside the container**, as defined above.

The executable command-line API uses the type of network (see [Network Configuration](#network-configuration) below) as the name of the executable to invoke. It will then look for this executable in a list of predefined directories. Once found, it will invoke the executable using the following environment variables for argument passing:
- `CNI_COMMAND`: indicates the desired operation; either `ADD` or `DEL`
- `CNI_CONTAINERID`: Container ID
- `CNI_NETNS`: Path to network namespace file
- `CNI_IFNAME`: Interface name to set up
- `CNI_ARGS`: Extra arguments passed in by the user at invocation time. Alphanumeric key-value pairs separated by semicolons; for example, "FOO=BAR;ABC=123"
- `CNI_PATH`: Colon-separated list of paths to search for CNI plugin executables

Network configuration in JSON format is streamed through stdin.


### Result

Success is indicated by a return code of zero and the following JSON printed to stdout in the case of the ADD command. This should be the same output as was returned by the IPAM plugin (see [IP Allocation](#ip-allocation) for details).

```
{
  "ip4": {
    "ip": <ipv4-and-subnet-in-CIDR>,
    "gateway": <ipv4-of-the-gateway>,  (optional)
    "routes": <list-of-ipv4-routes>    (optional)
  },
  "ip6": {
    "ip": <ipv6-and-subnet-in-CIDR>,
    "gateway": <ipv6-of-the-gateway>,  (optional)
    "routes": <list-of-ipv6-routes>    (optional)
  }
}
```

Errors are indicated by a non-zero return code and the following JSON being printed to stdout:
```
{
  "code": <numeric-error-code>,
  "msg": <short-error-message>,
  "details": <long-error-message> (optional)
}
```

Error codes 0-99 are reserved for well-known errors (to be defined later).
Values of 100+ can be freely used for plugin specific errors. 

In addition, stderr can be used for unstructured output such as logs.

### Network Configuration

The network configuration is described in JSON form. The configuration can be stored on disk or generated from other sources by the container runtime. The following fields are well-known and have the following meaning:
- `name` (string): Network name. This should be unique across all containers on the host (or other administrative domain).
- `type` (string): Refers to the filename of the CNI plugin executable.
- `ipMasq` (boolean): Optional (if supported by the plugin). Set up an IP masquerade on the host for this network. This is necessary if the host will act as a gateway to subnets that are not able to route to the IP assigned to the container.
- `ipam`: Dictionary with IPAM specific values:
  - `type` (string): Refers to the filename of the IPAM plugin executable.
  - `routes` (list): List of subnets (in CIDR notation) that the CNI plugin should ensure are reachable by routing them through the network. Each entry is a dictionary containing:
    - `dst` (string): subnet in CIDR notation
    - `gw` (string): IP address of the gateway to use. If not specified, the default gateway for the subnet is assumed (as determined by the IPAM plugin).

### Example configurations

```json
{
  "name": "dbnet",
  "type": "bridge",
  // type (plugin) specific
  "bridge": "cni0",
  "addIf": "eth0",
  "ipam": {
    "type": "host-local",
    // ipam specific
    "subnet": "10.1.0.0/16",
    "gateway": "10.1.0.1"
  },
}
```

```json
{
  "name": "pci",
  "type": "ovs",
  // type (plugin) specific
  "bridge": "ovs0",
  "vxlanID": 42,
  "ipam": {
    "type": "dhcp",
    "routes": [ { "dst": "10.3.0.0/16" }, { "dst": "10.4.0.0/16" } ]
  }
}
```

```json
{
  "name": "wan",
  "type": "macvlan",
  // ipam specific
  "ipam": {
    "type": "dhcp",
    "routes": [ { "dst": "10.0.0.0/8", "gw": "10.0.0.1" } ]
  },
}
```


### IP Allocation

As part of its operation, a CNI plugin is expected to assign (and maintain) an IP address to the interface and install any necessary routes relevant for that interface. This gives the CNI plugin great flexibility but also places a large burden on it. Many CNI plugins would need to have the same code to support several IP management schemes that users may desire (e.g. dhcp, host-local). 

To lessen the burden and make IP management strategy be orthogonal to the type of CNI plugin, we define a second type of plugin -- IP Address Management Plugin (IPAM plugin). It is however the responsibility of the CNI plugin to invoke the IPAM plugin at the proper moment in its execution. The IPAM plugin is expected to determine the interface IP/subnet, Gateway and Routes and return this information to the "main" plugin to apply. The IPAM plugin may obtain the information via a protocol (e.g. dhcp), data stored on a local filesystem, the "ipam" section of the Network Configuration file or a combination of the above.

#### IP Address Management (IPAM) Interface

Like CNI plugins, the IPAM plugins are invoked by running an executable. The executable is searched for in a predefined list of paths, indicated to the CNI plugin via `CNI_PATH`. The IPAM Plugin receives all the same environment variables that were passed in to the CNI plugin. Just like the CNI plugin, IPAM receives the network configuration file via stdin.

Success is indicated by a zero return code and the following JSON being printed to stdout (in the case of the ADD command):

```
{
  "ip4": {
    "ip": <ipv4-and-subnet-in-CIDR>,
    "gateway": <ipv4-of-the-gateway>,  (optional)
    "routes": <list-of-ipv4-routes>    (optional)
  },
  "ip6": {
    "ip": <ipv6-and-subnet-in-CIDR>,
    "gateway": <ipv6-of-the-gateway>,  (optional)
    "routes": <list-of-ipv6-routes>    (optional)
  }
}
```

`gateway` is the default gateway for this subnet, if one exists.
It does not instruct the CNI plugin to add any routes with this gateway: routes to add are specified separately via the `routes` field.
An example use of this value is for the CNI plugin to add this IP address to the linux-bridge to make it a gateway.

Each route entry is a dictionary with the following fields:
- `dst` (string): Destination subnet specified in CIDR notation.
- `gw` (string): IP of the gateway. If omitted, a default gateway is assumed (as determined by the CNI plugin).

Errors and logs are communicated in the same way as the CNI plugin. See [CNI Plugin Result](#result) section for details.

IPAM plugin examples:
 - **host-local**: Select an unused (by other containers on the same host) IP within the specified range.
 - **dhcp**: Use DHCP protocol to acquire and maintain a lease. The DHCP requests will be sent via the created container interface; therefore, the associated network must support broadcast.

#### Notes

 - Routes are expected to be added with a 0 metric.
 - A default route may be specified via "0.0.0.0/0". Since another network might have already configured the default route, the CNI plugin should be prepared to skip over its default route definition.

## Open Questions
- Should CNI define anything regarding DNS? For example, generating /etc/resolv.conf
- Should CNI provide /etc/hosts?
