# rkt Commands

Work in progress.
Please contribute if you see an area that needs more detail.

## Downloading Images (ACIs)

[aci-images]: https://github.com/appc/spec/blob/master/spec/aci.md#app-container-image
[appc-discovery]: https://github.com/appc/spec/blob/master/spec/discovery.md#app-container-image-discovery

rkt runs applications packaged according to the open-source [App Container Image][aci-images] specification.
ACIs consist of the root filesystem of the application container, a manifest, and an optional signature.

ACIs are named with a URL-like structure.
This naming scheme allows for a decentralized discovery of ACIs, related signatures and public keys.
rkt uses these hints to execute [meta discovery][appc-discovery].

* [trust](subcommands/trust.md)
* [fetch](subcommands/fetch.md)

## Running Pods

[metadata-spec]: https://github.com/appc/spec/blob/master/spec/ace.md#app-container-metadata-service
[rkt-mds]: subcommands/metadata-service.md

rkt can execute ACIs identified by name, hash, local file path, or URL.
If an ACI hasn't been cached on disk, rkt will attempt to find and download it.
To use rkt's [metadata service][metadata-spec], enable registration with the `--mds-register` flag when [invoking it][rkt-mds].

* [run](subcommands/run.md)
* [enter](subcommands/enter.md)
* [prepare](subcommands/prepare.md)
* [run-prepared](subcommands/run-prepared.md)

## Pod inspection and management

rkt provides subcommands to list, get status, and clean its pods.

* [list](subcommands/list.md)
* [status](subcommands/status.md)
* [gc](subcommands/gc.md)
* [rm](subcommands/rm.md)
* [cat-manifest](subcommands/cat-manifest.md)

## Interacting with the local image store

rkt provides subcommands to list, inspect and export images in its local store.

* [image](subcommands/image.md)

## Metadata Service

The metadata service helps running apps introspect their execution environment and assert their pod identity.

* [metadata-service](subcommands/metadata-service.md)

## API Service

The API service allows clients to list and inspect pods and images running under rkt.

* [api-service](subcommands/api-service.md)

## Misc

* [version](subcommands/version.md)

## Global Options

In addition to the flags used by individual `rkt` commands, `rkt` has a set of global options that are applicable to all commands.

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--debug` |  `false` | `true` or `false` | Prints out more debug information to `stderr` |
| `--dir` | `/var/lib/rkt` | A directory path | Path to the `rkt` data directory |
| `--insecure-options` |  none | **none**: All security features are enabled<br/>**http**: Allow HTTP connections. Be warned that this will send any credentials as clear text.<br/>**image**: Disables verifying image signatures<br/>**tls**: Accept any certificate from the server and any host name in that certificate<br/>**ondisk**: Disables verifying the integrity of the on-disk, rendered image before running. This significantly speeds up start time.<br/>**all**: Disables all security checks | Comma-separated list of security features to disable |
| `--local-config` |  `/etc/rkt` | A directory path | Path to the local configuration directory |
| `--system-config` |  `/usr/lib/rkt` | A directory path | Path to the system configuration directory |
| `--trust-keys-from-https` |  `false` | `true` or `false` | Automatically trust gpg keys fetched from https |
| `--user-config` |  `` | A directory path | Path to the user configuration directory |

## Logging

By default, rkt will send logs directly to stdout/stderr, allowing them to be captured by the invoking process.
On host systems running systemd, rkt will attempt to integrate with journald on the host.
In this case, the logs can be accessed directly via journalctl.

#### Accessing logs via journalctl

To read the logs of a running pod, get the pod's machine name from `machinectl`:

```
$ machinectl
MACHINE                                  CLASS     SERVICE
rkt-132f9d56-0e3f-4d1e-ba86-68efd488bb62 container nspawn

1 machines listed.
```

or `rkt list --full`

```
# rkt list --full
UUID					APP	IMAGE NAME		IMAGE ID		STATE	NETWORKS
132f9d56-0e3f-4d1e-ba86-68efd488bb62	etcd	coreos.com/etcd:v2.0.10 sha512-c03b055d02e5	running
```

The pod's machine name will be the pod's UUID prefixed with `rkt-`.
Given this machine name, logs can be retrieved by `journalctl`:

```
# journalctl -M rkt-132f9d56-0e3f-4d1e-ba86-68efd488bb62

[...]
```
