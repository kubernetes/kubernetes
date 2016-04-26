# govc

govc is a vSphere CLI built on top of govmomi.

## Installation

You can find prebuilt govc binaries on the [releases page](https://github.com/vmware/govmomi/releases).

Download and install a binary locally like this:

```sh
curl $URL_TO_BINARY | gzip -d > /usr/local/bin/govc
chmod +x /usr/local/bin/govc
```

### Source

You can install the latest govc version from source if you have the Go toolchain installed.

```sh
go get github.com/vmware/govmomi/govc
```

(make sure `$GOPATH/bin` is in your `PATH`)

## Usage

govc exposes its functionality through subcommands. Option flags
to these subcommands are often shared.

Common flags include:

* `-u`: ESXi or vCenter URL (ex: `user:pass@host`)
* `-debug`: Trace requests and responses (to `~/.govmomi/debug`)

Managed entities can be referred to by their absolute path or by their relative
path. For example, when specifying a datastore to use for a subcommand, you can
either specify it as `/mydatacenter/datastore/mydatastore`, or as
`mydatastore`. If you're not sure about the name of the datastore, or even the
full path to the datastore, you can specify a pattern to match. Both
`/*center/*/my*` (absolute) and `my*store` (relative) will resolve to the same
datastore, given there are no other datastores that match those globs.

The relative path in this example can only be used if the command can
umambigously resolve a datacenter to use as origin for the query. If no
datacenter is specified, govc defaults to the only datacenter, if there is only
one. The datacenter itself can be specified as a pattern as well, enabling the
following arguments: `-dc='my*' -ds='*store'`. The datastore pattern is looked
up and matched relative to the datacenter which itself is specified as a
pattern.

Besides specifying managed entities as arguments, they can also be specified
using environment variables. The following environment variables are used by govc
to set defaults:

* `GOVC_USERNAME`: USERNAME to use.

* `GOVC_PASSWORD`: PASSWORD to use.

* `GOVC_URL`: URL of ESXi or vCenter instance to connect to.

  > The URL scheme defaults to `https` and the URL path defaults to `/sdk`.
  > This means that specifying `user:pass@host` is equivalent to
  > `https://user:pass@host/sdk`.

  > If password include special characters like `#` or `:` you can use
  > `GOVC_USERNAME` and `GOVC_PASSWORD` to have a simple `GOVC_URL`

* `GOVC_INSECURE`: Allow establishing insecure connections.

  > Use this option when the host you're connecting is using self-signed
  > certificates, or is otherwise trusted. Set this option to `1` to enable.

* `GOVC_DATACENTER`

* `GOVC_DATASTORE`

* `GOVC_NETWORK`

* `GOVC_RESOURCE_POOL`

* `GOVC_HOST`

* `GOVC_GUEST_LOGIN`: Guest credentials for guest operations

* `GOVC_VIM_NAMESPACE`: Vim namespace defaults to `urn:vim25`

* `GOVC_VIM_VERSION`: Vim version defaults to `6.0`

## Examples

* About
  ```
  $ export GOVC_URL="192.168.1.20"
  $ export GOVC_USERNAME="domain\administrator"
  $ export GOVC_PASSWORD="Password123#"
  $ govc about

  Name:         VMware vCenter Server
  Vendor:       VMware, Inc.
  Version:      6.0.0
  Build:        2656761
  OS type:      linux-x64
  API type:     VirtualCenter
  API version:  6.0
  Product ID:   vpx
  UUID:         c9f0242f-10e3-4e10-85d7-5eea7c855188
  ```

* [Upload ssh public key to a VM](examples/lib/ssh.sh)

* [Create and configure a vCenter VM](examples/vcsa.sh)

## Projects using govc

* [Kubernetes vSphere Provider](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/cluster/vsphere)

* [Emacs govc package](./emacs)

## License

govc is available under the [Apache 2 license](../LICENSE).
