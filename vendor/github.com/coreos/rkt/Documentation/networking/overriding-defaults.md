# Overriding defaults

This document holds information about modifying or replacing builtin defaults and is only recommended to advanced users.
Please make sure to have read the [networking overview page][overview] before treading into these things.

## Overriding the "default" network

If a network has a name "default", it will override the default network added by rkt.
It is strongly recommended that such network also has type "ptp" as it protects from the pod spoofing its IP address and defeating identity management provided by the metadata service.

## Overriding network settings

The network backend CNI allows the passing of [arguments as plugin parameters][cni-plugin-parameters], specifically `CNI_ARGS`, at runtime.
These arguments can be used to reconfigure a network without changing the configuration file.
rkt supports the `CNI_ARGS` variable through the command line argument `--net`.

### Syntax

The syntax for passing arguments to a network looks like `--net="$networkname1:$arg1=$val1;$arg2=val2"`.
When executed from a shell, you can use double quotes to avoid `;` being interpreted as a command separator by the shell.
To allow the passing of arguments to different networks simply append the arguments to the network name with a colon (`:`), and separate the arguments by semicolon (`;`).
All arguments can either be given in a single instance of the `--net`, or can be spread across multiple uses of `--net`.
*Reminder:* the separator for the networks (and their arguments) within one `--net` instance is the comma `,`.
A network name must not be passed more than once, not within the same nor throughout multiple instances of `--net`.

### Example: Passing arguments to two different networks

This example will override the IP in the networks _net1_ and _net2_.

```bash
rkt run --net="net1:IP=1.2.3.4" --net="net2:IP=1.2.4.5" pod.aci
```

### Example: load all networks and override IPs for two different networks

This example will load all configured networks and override the IP addresses for *net1* and *net2*.

```bash
rkt run --net="all,net1:IP=1.2.3.4" --net="net2:IP=1.2.4.5" pod.aci
```

### Supported CNI\_ARGS

This is not documented yet.
Please follow CNI issue [#56][cni-56] to track the progress of the documentation.


[cni-56]: https://github.com/appc/cni/issues/56
[cni-plugin-parameters]: https://github.com/appc/cni/blob/master/SPEC.md#parameters
[overview]: overview.md
