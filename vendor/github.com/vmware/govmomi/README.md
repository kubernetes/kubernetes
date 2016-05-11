[![Build Status](https://travis-ci.org/vmware/govmomi.png?branch=master)](https://travis-ci.org/vmware/govmomi)
[![Build Status](https://ci.vmware.run/api/badges/vmware/govmomi/status.svg)](https://ci.vmware.run/vmware/govmomi)

# govmomi

A Go library for interacting with VMware vSphere APIs (ESXi and/or vCenter).

For `govc`, a CLI built on top of govmomi, check out the [govc](./govc) directory.

## Compatibility

This library is built for and tested against ESXi and vCenter 5.5 and 6.0.

If you're able to use it against older versions of ESXi and/or vCenter, please
leave a note and we'll include it in this compatibility list.

## Documentation

The APIs exposed by this library very closely follow the API described in the [VMware vSphere API Reference Documentation][apiref].
Refer to this document to become familiar with the upstream API.

The code in the `govmomi` package is a wrapper for the code that is generated from the vSphere API description.
It primarily provides convenience functions for working with the vSphere API.
See [godoc.org][godoc] for documentation.

[apiref]:http://pubs.vmware.com/vsphere-60/index.jsp#com.vmware.wssdk.apiref.doc/right-pane.html
[godoc]:http://godoc.org/github.com/vmware/govmomi
[drone]:https://drone.io
[dronesrc]:https://github.com/drone/drone
[dronecli]:http://readme.drone.io/devs/cli/

#### Building with CI
Merges to this repository will trigger builds in both Travis and [Drone][drone].

To build locally with Drone:
- Ensure that you have Docker 1.6 or higher installed.
- Install the [Drone command line tools][dronecli].
- Run `drone exec` from within the root directory of the govmomi repository.

## Status

Changes to the API are subject to [semantic versioning](http://semver.org).

Refer to the [CHANGELOG](CHANGELOG.md) for version to version changes.

## Projects using govmomi

* [Docker Machine](https://github.com/docker/machine/tree/master/drivers/vmwarevsphere)

* [Terraform](https://github.com/hashicorp/terraform/tree/master/builtin/providers/vsphere)

## License

govmomi is available under the [Apache 2 license](LICENSE).
