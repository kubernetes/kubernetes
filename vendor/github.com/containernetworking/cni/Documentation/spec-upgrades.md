# How to upgrade to CNI Specification v0.3.1

The 0.3.0 specification contained a small error. The Result structure's `ip` field should have been renamed to `ips` to be consistent with the IPAM result structure definition; this rename was missed when updating the Result to accommodate multiple IP addresses and interfaces. All first-party CNI plugins (bridge, host-local, etc) were updated to use `ips` (and thus be inconsistent with the 0.3.0 specification) and most other plugins have not been updated to the 0.3.0 specification yet, so few (if any) users should be impacted by this change.

The 0.3.1 specification corrects the Result structure to use the `ips` field name as originally intended.  This is the only change between 0.3.0 and 0.3.1.

# How to upgrade to CNI Specification v0.3.0

Version 0.3.0 of the [CNI Specification](../SPEC.md) provides rich information
about container network configuration, including details of network interfaces
and support for multiple IP addresses.

To support this new data, the specification changed in a couple significant
ways that will impact CNI users, plugin authors, and runtime authors.

This document provides guidance for how to upgrade:

- [For CNI Users](#for-cni-users)
- [For Plugin Authors](#for-plugin-authors)
- [For Runtime Authors](#for-runtime-authors)

**Note**: the CNI Spec is versioned independently from the GitHub releases
for this repo.  For example, Release v0.4.0 supports Spec version v0.2.0,
and Release v0.5.0 supports Spec v0.3.0.

----

## For CNI Users
If you maintain CNI configuration files for a container runtime that uses CNI,
ensure that the configuration files specify a `cniVersion` field and that the
version there is supported by your container runtime and CNI plugins.
Configuration files without a version field should be given version 0.2.0. 
The CNI spec includes example configuration files for 
[single plugins](https://github.com/containernetworking/cni/blob/master/SPEC.md#example-configurations)
and for [lists of chained plugins](https://github.com/containernetworking/cni/blob/master/SPEC.md#example-configurations).

Consult the documentation for your runtime and plugins to determine what
CNI spec versions they support. Test any plugin upgrades before deploying to 
production. You may find [cnitool](https://github.com/containernetworking/cni/tree/master/cnitool)
useful. Specifically, your configuration version should be the lowest common
version supported by your plugins.

## For Plugin Authors
This section provides guidance for upgrading plugins to CNI Spec Version 0.3.0.

### General guidance for all plugins (language agnostic)
To provide the smoothest upgrade path, **existing plugins should support
multiple versions of the CNI spec**.  In particular, plugins with existing
installed bases should add support for CNI spec version 0.3.0 while maintaining
compatibility with older versions.

To do this, two changes are required.  First, a plugin should advertise which
CNI spec versions it supports.  It does this by responding to the `VERSION`
command with the following JSON data:

```json
{
  "cniVersion": "0.3.0",
  "supportedVersions": [ "0.1.0", "0.2.0", "0.3.0" ]
}
```

Second, for the `ADD` command, a plugin must respect the `cniVersion` field
provided in the [network configuration JSON](https://github.com/containernetworking/cni/blob/master/SPEC.md#network-configuration). 
That field is a request for the plugin to return results of a particular format:

- If the `cniVersion` field is not present, then spec v0.2.0 should be assumed
	and v0.2.0 format result JSON returned.

- If the plugin doesn't support the version, the plugin must error.

- Otherwise, the plugin must return a [CNI Result](https://github.com/containernetworking/cni/blob/master/SPEC.md#result)
	in the format requested.

Result formats for older CNI spec versions are available in the
[git history for SPEC.md](https://github.com/containernetworking/cni/commits/master/SPEC.md).

For example, suppose a plugin, via its `VERSION` response, advertises CNI specification
support for v0.2.0 and v0.3.0.  When it receives `cniVersion` key of `0.2.0`,
the plugin must return result JSON conforming to CNI spec version 0.2.0.

### Specific guidance for plugins written in Go
Plugins written in Go may leverage the Go language packages in this repository
to ease the process of upgrading and supporting multiple versions.  CNI 
[Library and Plugins Release v0.5.0](https://github.com/containernetworking/cni/releases)
includes important changes to the Golang APIs.  Plugins using these APIs will
require some changes now, but should more-easily handle spec changes and
new features going forward.

For plugin authors, the biggest change is that `types.Result` is now an
interface implemented by concrete struct types in the `types/current` and
`types/020` subpackages.

Internally, plugins should use the `types/current` structs, and convert
to or from specific versions when required.  A typical plugin will only need
to do a single conversion.  That is when it is about to complete and needs to
print the result JSON in the correct format to stdout.  The library
function `types.PrintResult()` simplifies this by converting and printing in
a single call.

Additionally, the plugin should advertise which CNI Spec versions it supports
via the 3rd argument to `skel.PluginMain()`.

Here is some example code

```go
import (
	 "github.com/containernetworking/cni/pkg/skel"
	 "github.com/containernetworking/cni/pkg/types"
	 "github.com/containernetworking/cni/pkg/types/current"
	 "github.com/containernetworking/cni/pkg/version"
)

func cmdAdd(args *skel.CmdArgs) error {
	// determine spec version to use
	var netConf struct {
		types.NetConf
		// other plugin-specific configuration goes here
	}
	err := json.Unmarshal(args.StdinData, &netConf)
	cniVersion := netConf.CNIVersion

	// plugin does its work...
	//   set up interfaces
	//   assign addresses, etc
	
	// construct the result
	result := &current.Result{
		Interfaces: []*current.Interface{ ... },
		IPs: []*current.IPs{ ... },
		...
	}
	
	// print result to stdout, in the format defined by the requested cniVersion
	return types.PrintResult(result, cniVersion)
}

func main() {
	skel.PluginMain(cmdAdd, cmdDel, version.PluginSupports("0.1.0", "0.2.0", "0.3.0"))
}
```

Alternately, to use the result from a delegated IPAM plugin, the `result`
value might be formed like this:

```go
ipamResult, err := ipam.ExecAdd(netConf.IPAM.Type, args.StdinData)
result, err := current.NewResultFromResult(ipamResult)
```

Other examples of spec v0.3.0-compatible plugins are the
[main plugins in this repo](https://github.com/containernetworking/cni/tree/master/plugins/main)


## For Runtime Authors

This section provides guidance for upgrading container runtimes to support
CNI Spec Version 0.3.0.

### General guidance for all runtimes (language agnostic)

#### Support multiple CNI spec versions
To provide the smoothest upgrade path and support the broadest range of CNI
plugins, **container runtimes should support multiple versions of the CNI spec**.
In particular, runtimes with existing installed bases should add support for CNI
spec version 0.3.0 while maintaining compatibility with older versions.

To support multiple versions of the CNI spec, runtimes should be able to
call both new and legacy plugins, and handle the results from either.

When calling a plugin, the runtime must request that the plugin respond in a
particular format by specifying the `cniVersion` field in the
[Network Configuration](https://github.com/containernetworking/cni/blob/master/SPEC.md#network-configuration)
JSON block.  The plugin will then respond with
a [Result](https://github.com/containernetworking/cni/blob/master/SPEC.md#result)
in the format defined by that CNI spec version, and the runtime must parse
and handle this result.

#### Handle errors due to version incompatibility
Plugins may respond with error indicating that they don't support the requested
CNI version (see [Well-known Error Codes](https://github.com/containernetworking/cni/blob/master/SPEC.md#well-known-error-codes)),
e.g.
```json
{
  "cniVersion": "0.2.0",
  "code": 1,
  "msg": "CNI version not supported"
}
```
In that case, the runtime may retry with a lower CNI spec version, or take
some other action.

#### (optional) Discover plugin version support
Runtimes may discover which CNI spec versions are supported by a plugin, by
calling the plugin with the `VERSION` command.  The `VERSION` command was
added in CNI spec v0.2.0, so older plugins may not respect it.  In the absence
of a successful response to `VERSION`, assume that the plugin only supports
CNI spec v0.1.0.

#### Handle missing data in v0.3.0 results
The Result for the `ADD` command in CNI spec version 0.3.0 includes a new field
`interfaces`.  An IP address in the `ip` field may describe which interface
it is assigned to, by placing a numeric index in the `interface` subfield.

However, some plugins which are v0.3.0 compatible may nonetheless omit the
`interfaces` field and/or set the `interface` index value to `-1`.  Runtimes
should gracefully handle this situation, unless they have good reason to rely
on the existence of the interface data.  In that case, provide the user an
error message that helps diagnose the issue.

### Specific guidance for container runtimes written in Go
Container runtimes written in Go may leverage the Go language packages in this
repository to ease the process of upgrading and supporting multiple versions.
CNI [Library and Plugins Release v0.5.0](https://github.com/containernetworking/cni/releases)
includes important changes to the Golang APIs.  Runtimes using these APIs will
require some changes now, but should more-easily handle spec changes and
new features going forward.

For runtimes, the biggest changes to the Go libraries are in the `types` package.
It has been refactored to make working with versioned results simpler. The top-level 
`types.Result` is now an opaque interface instead of a struct, and APIs exposed by
other packages, such as the high-level `libcni` package, have been updated to use 
this interface.  Concrete types are now per-version subpackages. The `types/current`
subpackage contains the latest (spec v0.3.0) types.

When up-converting older result types to spec v0.3.0, fields new in
spec v0.3.0 (like `interfaces`) may be empty.  Conversely, when
down-converting v0.3.0 results to an older version, any data in those fields
will be lost.

| From   | 0.1 | 0.2 | 0.3 |
|--------|-----|-----|-----|
| To 0.1 |  ✔  |  ✔  |  x  |
| To 0.2 |  ✔  |  ✔  |  x  |
| To 0.3 |  ✴  |  ✴  |  ✔  |


Key:
> ✔ : lossless conversion <br>
> ✴ : higher-version output may have empty fields <br>
> x : lower-version output is missing some data <br>



A container runtime should use `current.NewResultFromResult()` to convert the
opaque  `types.Result` to a concrete `current.Result` struct.  It may then
work with the fields exposed by that struct:

```go
// runtime invokes the plugin to get the opaque types.Result
// this may conform to any CNI spec version
resultInterface, err := libcni.AddNetwork(netConf, runtimeConf)

// upconvert result to the current 0.3.0 spec
result, err := current.NewResultFromResult(resultInterface)

// use the result fields ....
for _, ip := range result.IPs { ... }
```
