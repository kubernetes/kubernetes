# Container Networking Interface Specification

## Version
This is CNI **spec** version **0.3.1**.

Note that this is **independent from the version of the CNI library and plugins** in this repository (e.g. the versions of [releases](https://github.com/containernetworking/cni/releases)).

## Overview

This document proposes a generic plugin-based networking solution for application containers on Linux, the _Container Networking Interface_, or _CNI_.
It is derived from the [rkt Networking Proposal][rkt-networking-proposal], which aimed to satisfy many of the [design considerations][rkt-networking-design] for networking in [rkt][rkt-github].

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

This document aims to specify the interface between "runtimes" and "plugins". Whilst there are certain well known fields, runtimes may wish to pass additional information to plugins. These extentions are not part of this specification but are documented as [conventions](CONVENTIONS.md).

## General considerations

The intention is for the container runtime to first create a new network namespace for the container.
It then determines which networks this container should belong to and for each network, which plugin must be executed.
The network configuration is in JSON format and can easily be stored in a file.
The network configuration includes mandatory fields such as "name" and "type" as well as plugin (type) specific ones.
The network configuration allows for fields to change values between invocations. For this purpose there is an optional field "args" which should contain the varying information.
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
    - **Version**. The version of CNI spec that the caller is using (container management system or the invoking plugin).
    - **Container ID**. This is optional but recommended, and should be unique across an administrative domain while the container is live (it may be reused in the future). For example, an environment with an IPAM system may require that each container is allocated a unique ID and that each IP allocation can thus be correlated back to a particular container. As another example, in appc implementations this would be the _pod ID_.
    - **Network namespace path**. This represents the path to the network namespace to be added, i.e. /proc/[pid]/ns/net or a bind-mount/link to it.
    - **Network configuration**. This is a JSON document describing a network to which a container can be joined. The schema is described below.
    - **Extra arguments**. This provides an alternative mechanism to allow simple configuration of CNI plugins on a per-container basis.
    - **Name of the interface inside the container**. This is the name that should be assigned to the interface created inside the container (network namespace); consequently it must comply with the standard Linux restrictions on interface names.
  - Result:
    - **Interfaces list**. Depending on the plugin, this can include the sandbox (eg, container or hypervisor) interface name and/or the host interface name, the hardware addresses of each interface, and details about the sandbox (if any) the interface is in.
    - **IP configuration assigned to each interface**. The IPv4 and/or IPv6 addresses, gateways, and routes assigned to sandbox and/or host interfaces.
    - **DNS information**. Dictionary that includes DNS information for nameservers, domain, search domains and options.

- Delete container from network
  - Parameters:
    - **Version**. The version of CNI spec that the caller is using (container management system or the invoking plugin).
    - **Container ID**, as defined above.
    - **Network namespace path**, as defined above.
    - **Network configuration**, as defined above.
    - **Extra arguments**, as defined above.
    - **Name of the interface inside the container**, as defined above.

- Report version
  - Parameters: NONE.
  - Result: information about the CNI spec versions supported by the plugin

      ```
      {
        "cniVersion": "0.3.1", // the version of the CNI spec in use for this output
        "supportedVersions": [ "0.1.0", "0.2.0", "0.3.0", "0.3.1" ] // the list of CNI spec versions that this plugin supports
      }
      ```

The executable command-line API uses the type of network (see [Network Configuration](#network-configuration) below) as the name of the executable to invoke.
It will then look for this executable in a list of predefined directories. Once found, it will invoke the executable using the following environment variables for argument passing:
- `CNI_COMMAND`: indicates the desired operation; `ADD`, `DEL` or `VERSION`.
- `CNI_CONTAINERID`: Container ID
- `CNI_NETNS`: Path to network namespace file
- `CNI_IFNAME`: Interface name to set up; plugin must honor this interface name or return an error
- `CNI_ARGS`: Extra arguments passed in by the user at invocation time. Alphanumeric key-value pairs separated by semicolons; for example, "FOO=BAR;ABC=123"
- `CNI_PATH`: List of paths to search for CNI plugin executables. Paths are separated by an OS-specific list separator; for example ':' on Linux and ';' on Windows

Network configuration in JSON format is streamed to the plugin through stdin. This means it is not tied to a particular file on disk and can contain information which changes between invocations.


### Result

Note that IPAM plugins return an abbreviated `Result` structure as described in [IP Allocation](#ip-allocation).

Success is indicated by a return code of zero and the following JSON printed to stdout in the case of the ADD command. The `ips` and `dns` items should be the same output as was returned by the IPAM plugin (see [IP Allocation](#ip-allocation) for details) except that the plugin should fill in the `interface` indexes appropriately, which are missing from IPAM plugin output since IPAM plugins should be unaware of interfaces.

```
{
  "cniVersion": "0.3.1",
  "interfaces": [                                            (this key omitted by IPAM plugins)
      {
          "name": "<name>",
          "mac": "<MAC address>",                            (required if L2 addresses are meaningful)
          "sandbox": "<netns path or hypervisor identifier>" (required for container/hypervisor interfaces, empty/omitted for host interfaces)
      }
  ],
  "ips": [
      {
          "version": "<4-or-6>",
          "address": "<ip-and-prefix-in-CIDR>",
          "gateway": "<ip-address-of-the-gateway>",          (optional)
          "interface": <numeric index into 'interfaces' list>
      },
      ...
  ],
  "routes": [                                                (optional)
      {
          "dst": "<ip-and-prefix-in-cidr>",
          "gw": "<ip-of-next-hop>"                           (optional)
      },
      ...
  ]
  "dns": {
    "nameservers": <list-of-nameservers>                     (optional)
    "domain": <name-of-local-domain>                         (optional)
    "search": <list-of-additional-search-domains>            (optional)
    "options": <list-of-options>                             (optional)
  }
}
```

`cniVersion` specifies a [Semantic Version 2.0](http://semver.org) of CNI specification used by the plugin.
`interfaces` describes specific network interfaces the plugin created.
If the `CNI_IFNAME` variable exists the plugin must use that name for the sandbox/hypervisor interface or return an error if it cannot.
- `mac` (string): the hardware address of the interface.
   If L2 addresses are not meaningful for the plugin then this field is optional.
- `sandbox` (string): container/namespace-based environments should return the full filesystem path to the network namespace of that sandbox.
   Hypervisor/VM-based plugins should return an ID unique to the virtualized sandbox the interface was created in.
   This item must be provided for interfaces created or moved into a sandbox like a network namespace or a hypervisor/VM.

The `ips` field is a list of IP configuration information.
See the [IP well-known structure](#ips) section for more information.

The `dns` field contains a dictionary consisting of common DNS information.
See the [DNS well-known structure](#dns) section for more information.

The specification does not declare how this information must be processed by CNI consumers.
Examples include generating an `/etc/resolv.conf` file to be injected into the container filesystem or running a DNS forwarder on the host.

Errors are indicated by a non-zero return code and the following JSON being printed to stdout:
```
{
  "cniVersion": "0.3.1",
  "code": <numeric-error-code>,
  "msg": <short-error-message>,
  "details": <long-error-message> (optional)
}
```

`cniVersion` specifies a [Semantic Version 2.0](http://semver.org) of CNI specification used by the plugin.
Error codes 0-99 are reserved for well-known errors (see [Well-known Error Codes](#well-known-error-codes) section).
Values of 100+ can be freely used for plugin specific errors. 

In addition, stderr can be used for unstructured output such as logs.

### Network Configuration

The network configuration is described in JSON form. The configuration can be stored on disk or generated from other sources by the container runtime. The following fields are well-known and have the following meaning:
- `cniVersion` (string): [Semantic Version 2.0](http://semver.org) of CNI specification to which this configuration conforms.
- `name` (string): Network name. This should be unique across all containers on the host (or other administrative domain).
- `type` (string): Refers to the filename of the CNI plugin executable.
- `args` (dictionary): Optional additional arguments provided by the container runtime. For example a dictionary of labels could be passed to CNI plugins by adding them to a labels field under `args`.
- `ipMasq` (boolean): Optional (if supported by the plugin). Set up an IP masquerade on the host for this network. This is necessary if the host will act as a gateway to subnets that are not able to route to the IP assigned to the container.
- `ipam`: Dictionary with IPAM specific values:
  - `type` (string): Refers to the filename of the IPAM plugin executable.
- `dns`: Dictionary with DNS specific values:
  - `nameservers` (list of strings): list of a priority-ordered list of DNS nameservers that this network is aware of. Each entry in the list is a string containing either an IPv4 or an IPv6 address.
  - `domain` (string): the local domain used for short hostname lookups.
  - `search` (list of strings): list of priority ordered search domains for short hostname lookups. Will be preferred over `domain` by most resolvers.
  - `options` (list of strings): list of options that can be passed to the resolver

Plugins may define additional fields that they accept and may generate an error if called with unknown fields. The exception to this is the `args` field may be used to pass arbitrary data which may be ignored by plugins.

### Example configurations

```json
{
  "cniVersion": "0.3.1",
  "name": "dbnet",
  "type": "bridge",
  // type (plugin) specific
  "bridge": "cni0",
  "ipam": {
    "type": "host-local",
    // ipam specific
    "subnet": "10.1.0.0/16",
    "gateway": "10.1.0.1"
  },
  "dns": {
    "nameservers": [ "10.1.0.1" ]
  }
}
```

```json
{
  "cniVersion": "0.3.1",
  "name": "pci",
  "type": "ovs",
  // type (plugin) specific
  "bridge": "ovs0",
  "vxlanID": 42,
  "ipam": {
    "type": "dhcp",
    "routes": [ { "dst": "10.3.0.0/16" }, { "dst": "10.4.0.0/16" } ]
  }
  // args may be ignored by plugins
  "args": {
    "labels" : {
        "appVersion" : "1.0"
    }
  }
}
```

```json
{
  "cniVersion": "0.3.1",
  "name": "wan",
  "type": "macvlan",
  // ipam specific
  "ipam": {
    "type": "dhcp",
    "routes": [ { "dst": "10.0.0.0/8", "gw": "10.0.0.1" } ]
  },
  "dns": {
    "nameservers": [ "10.0.0.1" ]
  }
}
```

### Network Configuration Lists

Network configuration lists provide a mechanism to run multiple CNI plugins for a single container in a defined order, passing the result of each plugin to the next plugin.
The list is composed of well-known fields and list of one or more standard CNI network configurations (see above).

The list is described in JSON form, and can be stored on disk or generated from other sources by the container runtime. The following fields are well-known and have the following meaning:
- `cniVersion` (string): [Semantic Version 2.0](http://semver.org) of CNI specification to which this configuration list and all the individual configurations conform.
- `name` (string): Network name. This should be unique across all containers on the host (or other administrative domain).
- `plugins` (list): A list of standard CNI network configuration dictionaries (see above).

When executing a plugin list, the runtime MUST replace the `name` and `cniVersion` fields in each individual network configuration in the list with the `name` and `cniVersion` field of the list itself. This ensures that the name and CNI version is the same for all plugin executions in the list, preventing versioning conflicts between plugins.
The runtime may also pass capability-based keys as a map in the top-level `runtimeConfig` key of the plugin's config JSON if a plugin advertises it supports a specific capability via the `capabilities` key of its network configuration.  The key passed in `runtimeConfig` MUST match the name of the specific capability from the `capabilities` key of the plugins network configuration. See CONVENTIONS.md for more information on capabilities and how they are sent to plugins via the `runtimeConfig` key.

For the ADD action, the runtime MUST also add a `prevResult` field to the configuration JSON of any plugin after the first one, which MUST be the Result of the previous plugin (if any) in JSON format ([see below](#network-configuration-list-runtime-examples)).
For the ADD action, plugins SHOULD echo the contents of the `prevResult` field to their stdout to allow subsequent plugins (and the runtime) to receive the result, unless they wish to modify or suppress a previous result.
Plugins are allowed to modify or suppress all or part of a `prevResult`.
However, plugins that support a version of the CNI specification that includes the `prevResult` field MUST handle `prevResult` by either passing it through, modifying it, or suppressing it explicitly.
It is a violation of this specification to be unaware of the `prevResult` field.

The runtime MUST also execute each plugin in the list with the same environment.

For the DEL action, the runtime MUST execute the plugins in reverse-order.

#### Network Configuration List Error Handling

When an error occurs while executing an action on a plugin list (eg, either ADD or DEL) the runtime MUST stop execution of the list.

If an ADD action fails, when the runtime decides to handle the failure it should execute the DEL action (in reverse order from the ADD as specified above) for all plugins in the list, even if some were not called during the ADD action.

Plugins should generally complete a DEL action without error even if some resources are missing.  For example, an IPAM plugin should generally release an IP allocation and return success even if the container network namespace no longer exists, unless that network namespace is critical for IPAM management. While DHCP may usually send a 'release' message on the container network interface, since DHCP leases have a lifetime this release action would not be considered critical and no error should be returned. For another example, the `bridge` plugin should delegate the DEL action to the IPAM plugin and clean up its own resources (if present) even if the container network namespace and/or container network interface no longer exist.

#### Example network configuration lists

```json
{
  "cniVersion": "0.3.1",
  "name": "dbnet",
  "plugins": [
    {
      "type": "bridge",
      // type (plugin) specific
      "bridge": "cni0",
      // args may be ignored by plugins
      "args": {
        "labels" : {
            "appVersion" : "1.0"
        }
      },
      "ipam": {
        "type": "host-local",
        // ipam specific
        "subnet": "10.1.0.0/16",
        "gateway": "10.1.0.1"
      },
      "dns": {
        "nameservers": [ "10.1.0.1" ]
      }
    },
    {
      "type": "tuning",
      "sysctl": {
        "net.core.somaxconn": "500"
      }
    }
  ]
}
```

#### Network configuration list runtime examples

Given the network configuration list JSON [shown above](#example-network-configuration-lists) the container runtime would perform the following steps for the ADD action.
Note that the runtime adds the `cniVersion` and `name` fields from configuration list to the configuration JSON passed to each plugin, to ensure consistent versioning and names for all plugins in the list.

1) first call the `bridge` plugin with the following JSON:

```json
{
  "cniVersion": "0.3.1",
  "name": "dbnet",
  "type": "bridge",
  "bridge": "cni0",
  "args": {
    "labels" : {
        "appVersion" : "1.0"
    }
  },
  "ipam": {
    "type": "host-local",
    // ipam specific
    "subnet": "10.1.0.0/16",
    "gateway": "10.1.0.1"
  },
  "dns": {
    "nameservers": [ "10.1.0.1" ]
  }
}
```

2) next call the `tuning` plugin with the following JSON, including the `prevResult` field containing the JSON response from the `bridge` plugin:

```json
{
  "cniVersion": "0.3.1",
  "name": "dbnet",
  "type": "tuning",
  "sysctl": {
    "net.core.somaxconn": "500"
  },
  "prevResult": {
    "ips": [
        {
          "version": "4",
          "address": "10.0.0.5/32",
          "interface": 0
        }
    ],
    "dns": {
      "nameservers": [ "10.1.0.1" ]
    }
  }
}
```

Given the same network configuration JSON list, the container runtime would perform the following steps for the DEL action.
Note that no `prevResult` field is required as the DEL action does not return any result.
Also note that plugins are executed in reverse order from the ADD action.

1) first call the `tuning` plugin with the following JSON:

```json
{
  "cniVersion": "0.3.1",
  "name": "dbnet",
  "type": "tuning",
  "sysctl": {
    "net.core.somaxconn": "500"
  }
}
```

2) next call the `bridge` plugin with the following JSON:

```json
{
  "cniVersion": "0.3.1",
  "name": "dbnet",
  "type": "bridge",
  "bridge": "cni0",
  "args": {
    "labels" : {
        "appVersion" : "1.0"
    }
  },
  "ipam": {
    "type": "host-local",
    // ipam specific
    "subnet": "10.1.0.0/16",
    "gateway": "10.1.0.1"
  },
  "dns": {
    "nameservers": [ "10.1.0.1" ]
  }
}
```

### IP Allocation

As part of its operation, a CNI plugin is expected to assign (and maintain) an IP address to the interface and install any necessary routes relevant for that interface. This gives the CNI plugin great flexibility but also places a large burden on it. Many CNI plugins would need to have the same code to support several IP management schemes that users may desire (e.g. dhcp, host-local). 

To lessen the burden and make IP management strategy be orthogonal to the type of CNI plugin, we define a second type of plugin -- IP Address Management Plugin (IPAM plugin). It is however the responsibility of the CNI plugin to invoke the IPAM plugin at the proper moment in its execution. The IPAM plugin is expected to determine the interface IP/subnet, Gateway and Routes and return this information to the "main" plugin to apply. The IPAM plugin may obtain the information via a protocol (e.g. dhcp), data stored on a local filesystem, the "ipam" section of the Network Configuration file or a combination of the above.

#### IP Address Management (IPAM) Interface

Like CNI plugins, the IPAM plugins are invoked by running an executable. The executable is searched for in a predefined list of paths, indicated to the CNI plugin via `CNI_PATH`. The IPAM Plugin receives all the same environment variables that were passed in to the CNI plugin. Just like the CNI plugin, IPAM receives the network configuration via stdin.

Success is indicated by a zero return code and the following JSON being printed to stdout (in the case of the ADD command):

```
{
  "cniVersion": "0.3.1",
  "ips": [
      {
          "version": "<4-or-6>",
          "address": "<ip-and-prefix-in-CIDR>",
          "gateway": "<ip-address-of-the-gateway>"  (optional)
      },
      ...
  ],
  "routes": [                                       (optional)
      {
          "dst": "<ip-and-prefix-in-cidr>",
          "gw": "<ip-of-next-hop>"                  (optional)
      },
      ...
  ]
  "dns": {
    "nameservers": <list-of-nameservers>            (optional)
    "domain": <name-of-local-domain>                (optional)
    "search": <list-of-search-domains>              (optional)
    "options": <list-of-options>                    (optional)
  }
}
```

Note that unlike regular CNI plugins, IPAM plugins return an abbreviated `Result` structure that does not include the `interfaces` key, since IPAM plugins should be unaware of interfaces configured by their parent plugin except those specifically required for IPAM (eg, like the `dhcp` IPAM plugin).

`cniVersion` specifies a [Semantic Version 2.0](http://semver.org) of CNI specification used by the plugin.

The `ips` field is a list of IP configuration information.
See the [IP well-known structure](#ips) section for more information.

The `dns` field contains a dictionary consisting of common DNS information.
See the [DNS well-known structure](#dns) section for more information.

Errors and logs are communicated in the same way as the CNI plugin. See [CNI Plugin Result](#result) section for details.

IPAM plugin examples:
 - **host-local**: Select an unused (by other containers on the same host) IP within the specified range.
 - **dhcp**: Use DHCP protocol to acquire and maintain a lease. The DHCP requests will be sent via the created container interface; therefore, the associated network must support broadcast.

#### Notes
 - Routes are expected to be added with a 0 metric.
 - A default route may be specified via "0.0.0.0/0". Since another network might have already configured the default route, the CNI plugin should be prepared to skip over its default route definition.

### Well-known Structures

#### IPs

```
  "ips": [
      {
          "version": "<4-or-6>",
          "address": "<ip-and-prefix-in-CIDR>",
          "gateway": "<ip-address-of-the-gateway>",      (optional)
          "interface": <numeric index into 'interfaces' list> (not required for IPAM plugins)
      },
      ...
  ]
```

The `ips` field is a list of IP configuration information determined by the plugin. Each item is a dictionary describing of IP configuration for a network interface.
IP configuration for multiple network interfaces and multiple IP configurations for a single interface may be returned as separate items in the `ips` list.
All properties known to the plugin should be provided, even if not strictly required.
- `version` (string): either "4" or "6" and corresponds to the IP version of the addresses in the entry.
   All IP addresses and gateways provided must be valid for the given `version`.
- `address` (string): an IP address in CIDR notation (eg "192.168.1.3/24").
- `gateway` (string): the default gateway for this subnet, if one exists.
   It does not instruct the CNI plugin to add any routes with this gateway: routes to add are specified separately via the `routes` field.
   An example use of this value is for the CNI `bridge` plugin to add this IP address to the Linux bridge to make it a gateway.
- `interface` (uint): the index into the `interfaces` list for a [CNI Plugin Result](#result) indicating which interface this IP configuration should be applied to.
   IPAM plugins should not return this key since they have no information about network interfaces.

#### Routes

```
  "routes": [
      {
          "dst": "<ip-and-prefix-in-cidr>",
          "gw": "<ip-of-next-hop>"               (optional)
      },
      ...
  ]
```

- Each `routes` entry is a dictionary with the following fields.  All IP addresses in the `routes` entry must be the same IP version, either 4 or 6.
  - `dst` (string): destination subnet specified in CIDR notation.
  - `gw` (string): IP of the gateway. If omitted, a default gateway is assumed (as determined by the CNI plugin).

#### DNS

```
  "dns": {
    "nameservers": <list-of-nameservers>                 (optional)
    "domain": <name-of-local-domain>                     (optional)
    "search": <list-of-additional-search-domains>        (optional)
    "options": <list-of-options>                         (optional)
  }
```

The `dns` field contains a dictionary consisting of common DNS information.
- `nameservers` (list of strings): list of a priority-ordered list of DNS nameservers that this network is aware of. Each entry in the list is a string containing either an IPv4 or an IPv6 address.
- `domain` (string): the local domain used for short hostname lookups.
- `search` (list of strings): list of priority ordered search domains for short hostname lookups. Will be preferred over `domain` by most resolvers.
- `options` (list of strings): list of options that can be passed to the resolver.
  See [CNI Plugin Result](#result) section for more information.

## Well-known Error Codes
- `1` - Incompatible CNI version
- `2` - Unsupported field in network configuration. The error message must contain the key and value of the unsupported field.
