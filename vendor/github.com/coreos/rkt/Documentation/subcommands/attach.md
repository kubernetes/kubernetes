# rkt attach

Applications can be started in interactive mode and later attached via `rkt attach`.

In order for an application to be attachable:
 * it must be started in interactive mode
 * it must be running as part of a running pod
 * it must support the corresponding attach mode

To start an application in interactive mode, either `tty` or `stream` must be passed as value for the `--stdin`, `--stdout` and `--stderr` options.

An application can be run with a dedicated terminal and later attached to:

```
# rkt run quay.io/coreos/alpine-sh --stdin tty --stdout tty --stderr tty
```

```
# rkt attach --mode tty ${UUID}

/ # hostname
rkt-911afe8e-992f-4089-8666-4a4c957a1964

/ # tty     
/rkt/iottymux/alpine-sh/pts
^C
```

In a similar way, an application can be run without a tty but with separated attachable streams:

```
# rkt run quay.io/coreos/alpine-sh --stdin stream --stdout stream --stderr stream
```

```
# rkt attach --mode stdin,stdout,stderr ${UUID}
hostname
rkt-846c35db-6728-471a-ad50-66d3a8d7ff9c

tty
not a tty
^C
```

If a pod contains multiple applications, the one to be used as attach target can be specified via `--app`.

The following options are allowed as `--mode` values:
 * `list`: list available endpoints, and return early without attaching
 * `auto`: attach to all available endpoints
 * `tty`: bi-directionally attach to the application terminal
 * `tty-in` or `tty-out`: uni-directionally attach to the application terminal
 * `stdin,stdout,stderr`: attach to specific application streams. Omitted streams will no be attached

A more complex example, showing the usage of advanced options and piping:

```
# rkt run quay.io/coreos/alpine-sh --stdin stream --stdout stream --stderr stream
```

```
# rkt attach --app alpine-sh --mode list 846c35db
stdin
stdout
stderr

# echo 'hostname; fakecmd' | ./rkt attach --app alpine-sh --mode auto ${UUID}
rkt-846c35db-6728-471a-ad50-66d3a8d7ff9c
/bin/sh: fakecmd: not found
^C

# echo 'hostname; fakecmd' | ./rkt attach --app alpine-sh --mode stdin,stdout ${UUID}
rkt-846c35db-6728-471a-ad50-66d3a8d7ff9c
^C

# echo 'hostname; fakecmd' | ./rkt attach --app alpine-sh --mode stdin,stderr ${UUID}
/bin/sh: fakecmd: not found
^C
```

## Options

| Flag | Default | Options | Description |
| ---  | ---     | ---     | ---         |
| `--app`  |  ``     | Name of an application            | Name of the app to attach to within the specified pod |
| `--mode` |  `auto` | "list", "auto" or tty/stream mode | Attaching mode                                        |

## Global options

See the table with [global options in general commands documentation][global-options].


[global-options]: ../commands.md#global-options
