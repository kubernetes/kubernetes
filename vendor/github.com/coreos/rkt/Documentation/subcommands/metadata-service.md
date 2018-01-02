# rkt metadata-service

## Overview

The metadata service is designed to help running apps introspect their execution environment and assert their pod identity.
In particular, the metadata service exposes the contents of the pod and image manifests as well as a convenient method of looking up annotations.
Finally, the metadata service provides a pod with cryptographically verifiable identity.

## Running the metadata service

The metadata service is implemented by the `rkt metadata-service` command.
When started, it will listen for registration events over Unix socket on `/run/rkt/metadata-svc.sock`.
For systemd-based distributions, it also supports [systemd socket activation][socket-activation].

If using socket activation, ensure the socket is named `/run/rkt/metadata-svc.sock`, as `rkt run` uses this name during registration.
Please note that when started under socket activation, the metadata service will not remove the socket on exit.
Use the `RemoveOnStop` directive in the relevant `.socket` file to clean up.

Example systemd unit files for running the metadata service are available in [dist][dist].

In addition to listening on a Unix socket, the metadata service will also listen on a TCP port 2375.
When contacting the metadata service, the apps utilize this port.
The IP and port of the metadata service are passed by rkt to pods via the `AC_METADATA_URL` environment variable.

## Using the metadata service

See [App Container specification][appc-container-metadata] for more information about the metadata service including a list of supported endpoints and their usage.

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--listen-port` |  `18112` | A port number | Listen port |

## Global options

See the table with [global options in general commands documentation][global-options].


[appc-container-metadata]: https://github.com/appc/spec/blob/master/spec/ace.md#app-container-metadata-service
[dist]: https://github.com/coreos/rkt/tree/master/dist/init/systemd
[global-options]: ../commands.md#global-options
[socket-activation]: http://0pointer.de/blog/projects/socket-activation.html
