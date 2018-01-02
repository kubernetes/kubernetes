# Logging and attaching design

## Overview

rkt can run multiple applications in a pod, under a supervising process and alongside with a sidecar service which takes care of multiplexing its I/O toward the outside world.

Historically this has been done via systemd-journald only, meaning that all logging was handled via journald and interactive applications had to re-use a parent TTY.

Starting from systemd v232, it is possible to connect a service streams to arbitrary socket units and let custom sidecar multiplex all the I/O.

This document describes the architectural design for the current logging and attaching subsystem, which allows custom logging and attaching logic.

## Runtime modes

In order to be able to attach or apply custom logging logic to applications, an appropriate runtime mode must be specified when adding/preparing an application inside a pod.

This is done via stage0 CLI arguments (`--stdin`, `--stdout`, and `--stder`) which translate into per-application stage2 annotations.

### Interactive mode

This mode results in the application having the corresponding stream attached to the parent terminal.

For historical reasons and backward compatibility, this is a special mode activated via `--interactive` and only supports single-app pods.

Interactive mode does not support attaching and ties the runtime to the lifetime of the parent terminal.

Internally, this translates to an annotation at the app level:

```
{
    "name": "coreos.com/rkt/stage2/stdin",
    "value": "interactive"
},
{
    "name": "coreos.com/rkt/stage2/stdout",
    "value": "interactive"
},
{
    "name": "coreos.com/rkt/stage2/stderr",
    "value": "interactive"
}
```

In this case, the corresponding service unit file gains the following properties:

```
[Service]
StandardInput=tty
StandardOutput=tty
StandardError=tty
...
```

No further sidecar dependencies are introduced in this case.

### TTY mode

This mode results in the application having the corresponding stream attached to a dedicated pseudo-terminal.

This is different from the "interactive" mode because:
 * it allocates a new pseudo-terminal accounted towards pod resources
 * it supports external attaching/detaching
 * it supports multiple applications running inside a single pod
 * it does not tie the pod lifetime to the parent terminal one

Internally, this translates to an annotation at the app level:

```
{
    "name": "coreos.com/rkt/stage2/stdin",
    "value": "tty"
},
{
    "name": "coreos.com/rkt/stage2/stdout",
    "value": "tty"
},
{
    "name": "coreos.com/rkt/stage2/stderr",
    "value": "tty"
}
```

In this case, the corresponding service unit file gains the following properties:

```
[Service]
TTYPath=/rkt/iomux/<appname>/stage2-pts
StandardInput=tty
StandardOutput=tty
StandardError=tty
...
```

A sidecar dependency to `ttymux@.service` is introduced in this case. Application has a `Wants=` and `After=` relationship to it.

### Streaming mode

This mode results in the application having each of the corresponding streams separately handled by a muxing service.

This is different from the "interactive" and "tty" modes because:
 * it does not allocate any terminal for the application
 * single streams can be separately handled
 * it supports multiple applications running inside a single pod


Internally, this translates to an annotation at the app level:

```
{
    "name": "coreos.com/rkt/stage2/stdin",
    "value": "stream"
},
{
    "name": "coreos.com/rkt/stage2/stdout",
    "value": "stream"
},
{
    "name": "coreos.com/rkt/stage2/stderr",
    "value": "stream"
}
```

In this case, the corresponding service unit file gains the following properties:

```
[Service]
StandardInput=fd
Sockets=<appname>-stdin.socket
StandardOutput=fd
Sockets=<appname>-stdout.socket
StandardError=fd
Sockets=<appname>-stderr.socket
...
```

A sidecar dependency to `iomux@.service` is introduced in this case. Application has a `Wants=` and `Before=` relationship to it.

Additional per-stream socket units are generated, as follows:

```
[Unit]
Description=<stream> socket for <appname>
DefaultDependencies=no
StopWhenUnneeded=yes
RefuseManualStart=yes
RefuseManualStop=yes
BindsTo=<appname>.service

[Socket]
RemoveOnStop=yes
Service=<appname>.service
FileDescriptorName=<stream>
ListenFIFO=/rkt/iottymux/<appname>/stage2-<stream>
```

### Logging mode

This mode results in the application having the corresponding stream attached to systemd-journald.

This is the default mode for stdout/stderr, for historical reasons and backward compatibility.

Internally, this translates to an annotation at the app level:

```
{
    "name": "coreos.com/rkt/stage2/stdout",
    "value": "log"
},
{
    "name": "coreos.com/rkt/stage2/stderr",
    "value": "log"
}
```

In this case, the corresponding service unit file gains the following properties:

```
[Service]
StandardOutput=journal
StandardError=journal
...
```

A sidecar dependency to `systemd-journald.service` is introduced in this case. Application has a `Wants=` and `After=` relationship to it.

Logging is not a valid mode for stdin.

### Null mode

This mode results in the application having the corresponding stream closed.

This is the default mode for stdin, for historical reasons and backward compatibility.

Internally, this translates to an annotation at the app level:

```
{
    "name": "coreos.com/rkt/stage2/stdin",
    "value": "null"
},
{
    "name": "coreos.com/rkt/stage2/stdout",
    "value": "null"
},
{
    "name": "coreos.com/rkt/stage2/stderr",
    "value": "null"
}
```

In this case, the corresponding service unit file gains the following properties:

```
[Service]
StandardInput=null
StandardOutput=null
StandardError=null
[...]
```

No further sidecar dependencies are introduced in this case.

## Annotations

The following per-app annotations are defined for internal use, with the corresponding set of allowed values:

 * `coreos.com/rkt/stage2/stdin`
   - `interactive`
   - `null`
   - `stream`
   - `tty`
 * `coreos.com/rkt/stage2/stdout`
   - `interactive`
   - `log`
   - `null`
   - `stream`
   - `tty`
 * `coreos.com/rkt/stage2/stderr`
   - `interactive`
   - `log`
   - `null`
   - `stream`
   - `tty`


## Stage1 internals

All the logging and attaching logic is handled by the stage1 `iottymux` binary.

Each main application may additionally have a dedicated sidecar for I/O multiplexing, which proxies I/O to external clients over sockets.

Sidecar state is persisted at `/rkt/iottymux/<appname>` while the main application is running.


## Attaching

`rkt attach` can auto-discover endpoints, by reading the content of status file located at `/rkt/iottymux/<appname>/endpoints`.

This file provides a versioned JSON document, whose content varies depending on the I/O for the specific application.

For example, an application with all streams available for attaching will have a status file similar to the following:

```
{
    "version": 1,
    "targets": [
        {
            "name": "stdin",
            "domain": "unix",
            "address": "/rkt/iottymux/alpine-sh/sock-stdin"
        },
        {
            "name": "stdout",
            "domain": "unix",
            "address": "/rkt/iottymux/alpine-sh/sock-stdout"
        },
        {
            "name": "stderr",
            "domain": "unix",
            "address": "/rkt/iottymux/alpine-sh/sock-stderr"
        }
    ]
}
```

### Endpoint listing


Its `--mode=list` option just read the file and print it back to the user.

### Automatic attaching

`rkt attach --mode=auto` performs the auto-discovery mechanism described above, and the proceed to attach stdin/stdour/stderr of the current process (itself) to all available corresponding endpoints.

This the default attaching mode.

### Custom attaching

`rkt attach --mode=<stream>` performs the auto-discovery mechanism described above, and the proceed to the corresponding available endpoints.

## Logging

### Journald

This is the default output multiplexer for stdout/stderr in logging mode, for historical reasons and backward compatibility.

Restrictions:
 * requires journalctl (or similar libsystemd-based helper) to decode output entries
 * requires a libsystemd on the host compiled with LZ4 support
 * systemd-journald does not support distinguishing between entries from stdout and stderr

### Experimental logging modes

TODO(lucab): k8s logmode

## Sidecars

### systemd-journald


This is the standard systemd-journald service. It is the default output handler for the "logging" mode.

### iottymux

iottymux is a multi-purpose stage1 binary. It currently serves the following purposes:
 * Multiplex I/O over TTY (in TTY mode)
 * Multiplex I/O from streams (in streaming mode)
 * Attach to existing attachable applications (in TTY or streaming mode)

#### iomux

This component takes care of multiplexing dedicated streams and receiving clients for attaching.

It is started as an instance of the templated `iomux@.service` service by a `Before=` dependency from the application.

Internally, it attaches to available FIFOs and proxies them to separate sockets for external clients.

It is implemented as a sub-action of the main `iottymux` binary and completely run in stage1 context.

#### ttymux

This component takes care of multiplexing TTY and receiving clients for attaching.

It is started as an instance of the templated `ttymux@.service` service by a `After=` dependency from the application.

Internally, it creates a pesudo-tty pair (whose slave is used by the main application) and proxies the master to a socket for external clients.

It is implemented as a sub-action of the main `iottymux` binary and completely run in stage1 context.

#### iottymux-attach

This component takes care of discovering endpoints and attaching to them, both for TTY and streaming modes.

It is invoked by the "stage1" attach entrypoint and completely run in stage1 context. It is implemented as a sub-action of the main `iottymux` binary.
