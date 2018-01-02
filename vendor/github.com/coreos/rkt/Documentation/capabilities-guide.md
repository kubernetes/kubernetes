# Capabilities Isolators Guide

This document is a walk-through guide describing how to use rkt isolators for
[Linux Capabilities][capabilities].

* [About Linux Capabilities](#about-linux-capabilities)
* [Default Capabilities](#default-capabilities)
* [Capability Isolators](#capability-isolators)
* [Usage Example](#usage-example)
* [Overriding Capabilities](#overriding-capabilities)
* [Recommendations](#recommendations)

## About Linux Capabilities

Linux capabilities are meant to be a modern evolution of traditional UNIX
permissions checks.
The goal is to split the permissions granted to privileged processes into a set
of capabilities (eg. `CAP_NET_RAW` to open a raw socket), which can be
separately handled and assigned to single threads.

Processes can gain specific capabilities by either being run by superuser, or by
having the setuid/setgid bits or specific file-capabilities set on their
executable file.
Once running, each process has a bounding set of capabilities which it can
enable and use; such process cannot get further capabilities outside of this set.

In the context of containers, capabilities are useful for:

* Restricting the effective privileges of applications running as root
* Allowing applications to perform specific privileged operations, without
   having to run them as root

For the complete list of existing Linux capabilities and a detailed description
of this security mechanism, see the [capabilities(7) man page][man-capabilities].

## Default capabilities

By default, rkt enforces [a default set of capabilities][default-caps] onto applications.
This default set is tailored to stop applications from performing a large
variety of privileged actions, while not impacting their normal behavior.
Operations which are typically not needed in containers and which may
impact host state, eg. invoking `reboot(2)`, are denied in this way.

However, this default set is mostly meant as a safety precaution against erratic
and misbehaving applications, and will not suffice against tailored attacks.
As such, it is recommended to fine-tune the capabilities bounding set using one
of the customizable isolators available in rkt.

## Capability Isolators

When running Linux containers, rkt provides two mutually exclusive isolators
to define the bounding set under which an application will be run:

* `os/linux/capabilities-retain-set`
* `os/linux/capabilities-remove-set`

Those isolators cover different use-cases and employ different techniques to
achieve the same goal of limiting available capabilities. As such, they cannot
be used together at the same time, and recommended usage varies on a
case-by-case basis.

As the granularity of capabilities varies for specific permission cases, a word
of warning is needed in order to avoid a false sense of security.
In many cases it is possible to abuse granted capabilities in order to
completely subvert the sandbox: for example, `CAP_SYS_PTRACE` allows to access
stage1 environment and `CAP_SYS_ADMIN` grants a broad range of privileges,
effectively equivalent to root.
Many other ways to maliciously transition across capabilities have already been
[reported][grsec-forums].

### Retain-set

`os/linux/capabilities-retain-set` allows for an additive approach to
capabilities: applications will be stripped of all capabilities, except the ones
listed in this isolator.

This whitelisting approach is useful for completely locking down environments
and whenever application requirements (in terms of capabilities) are
well-defined in advance. It allows one to ensure that exactly and only the
specified capabilities could ever be used.

For example, an application that will only need to bind to port 80 as
a privileged operation, will have `CAP_NET_BIND_SERVICE` as the only entry in
its "retain-set".

### Remove-set

`os/linux/capabilities-remove-set` tackles capabilities in a subtractive way:
starting from the default set of capabilities, single entries can be further
forbidden in order to prevent specific actions.

This blacklisting approach is useful to somehow limit applications which have
broad requirements in terms of privileged operations, in order to deny some
potentially malicious operations.

For example, an application that will need to perform multiple privileged
operations but is known to never open a raw socket, will have
`CAP_NET_RAW` specified in its "remove-set".

## Usage Example

The goal of these examples is to show how to build ACIs with [`acbuild`][acbuild],
where some capabilities are either explicitly blocked or allowed.
For simplicity, the starting point will be the official Alpine Linux image from
CoreOS which ships with `ping` and `nc` commands (from busybox). Those
commands respectively requires `CAP_NET_RAW` and `CAP_NET_BIND_SERVICE`
capabilities in order to perform privileged operations.
To block their usage, capabilities bounding set
can be manipulated via `os/linux/capabilities-remove-set` or
`os/linux/capabilities-retain-set`; both approaches are shown here.

### Removing specific capabilities

This example shows how to block `ping` only, by removing `CAP_NET_RAW` from
capabilities bounding set.

First, a local image is built with an explicit "remove-set" isolator.
This set contains the capabilities that need to be forbidden in order to block
`ping` usage (and only that):

```
$ acbuild begin
$ acbuild set-name localhost/caps-remove-set-example
$ acbuild dependency add quay.io/coreos/alpine-sh
$ acbuild set-exec -- /bin/sh
$ echo '{ "set": ["CAP_NET_RAW"] }' | acbuild isolator add "os/linux/capabilities-remove-set" -
$ acbuild write caps-remove-set-example.aci
$ acbuild end
```

Once properly built, this image can be run in order to check that `ping` usage has
been effectively disabled:

```
$ sudo rkt run --interactive --insecure-options=image caps-remove-set-example.aci
image: using image from file stage1-coreos.aci
image: using image from file caps-remove-set-example.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # whoami
root

/ # ping -c 1 8.8.8.8
PING 8.8.8.8 (8.8.8.8): 56 data bytes
ping: permission denied (are you root?)
```

This means that `CAP_NET_RAW` had been effectively disabled inside the container.
At the same time, `CAP_NET_BIND_SERVICE` is still available in the default bounding
set, so the `nc` command will be able to bind to port 80:

```
$ sudo rkt run --interactive --insecure-options=image caps-remove-set-example.aci
image: using image from file stage1-coreos.aci
image: using image from file caps-remove-set-example.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # whoami
root

/ # nc -v -l -p 80
listening on [::]:80 ...
```

### Allowing specific capabilities

In contrast to the example above, this one shows how to allow `ping` only, by
removing all capabilities except `CAP_NET_RAW` from the bounding set.
This means that all other privileged operations, including binding to port 80
will be blocked.

First, a local image is built with an explicit "retain-set" isolator.
This set contains the capabilities that need to be enabled in order to allowed
`ping` usage (and only that):

```
$ acbuild begin
$ acbuild set-name localhost/caps-retain-set-example
$ acbuild dependency add quay.io/coreos/alpine-sh
$ acbuild set-exec -- /bin/sh
$ echo '{ "set": ["CAP_NET_RAW"] }' | acbuild isolator add "os/linux/capabilities-retain-set" -
$ acbuild write caps-retain-set-example.aci
$ acbuild end
```

Once run, it can be easily verified that `ping` from inside the container is now
functional:

```
$ sudo rkt run --interactive --insecure-options=image caps-retain-set-example.aci
image: using image from file stage1-coreos.aci
image: using image from file caps-retain-set-example.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # whoami
root

/ # ping -c 1 8.8.8.8
PING 8.8.8.8 (8.8.8.8): 56 data bytes
64 bytes from 8.8.8.8: seq=0 ttl=41 time=24.910 ms

--- 8.8.8.8 ping statistics ---
1 packets transmitted, 1 packets received, 0% packet loss
round-trip min/avg/max = 24.910/24.910/24.910 ms
```

However, all others capabilities are now not anymore available to the application.
For example, using `nc` to bind to port 80 will now result in a failure due to
the missing `CAP_NET_BIND_SERVICE` capability:

```
$ sudo rkt run --interactive --insecure-options=image caps-retain-set-example.aci
image: using image from file stage1-coreos.aci
image: using image from file caps-retain-set-example.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # whoami
root

/ # nc -v -l -p 80
nc: bind: Permission denied
```

## Overriding capabilities

Capability sets are typically defined when creating images, as they are tightly
linked to specific app requirements. However, image consumers may need to further
tweak/restrict the set of available capabilities in specific local scenarios.
This can be done either by permanently patching the manifest of specific images, or
by overriding capability isolators with command line options.

### Patching images

Image manifests can be manipulated manually, by unpacking the image and editing
the manifest file, or with helper tools like [`actool`][actool].
To override an image's pre-defined capabilities set, replace the existing capabilities
isolators in the image with new isolators defining the desired capabilities.

The `patch-manifest` subcommand to `actool` manipulates the capabilities sets
defined in an image.
`actool patch-manifest --capability` changes the `retain` capabilities set.
`actool patch-manifest --revoke-capability` changes the `remove` set.
These commands take an input image, modify its existing capabilities sets, and
write the changes to an output image, as shown in the example:

```
$ actool cat-manifest caps-retain-set-example.aci
...
    "isolators": [
      {
        "name": "os/linux/capabilities-retain-set",
        "value": {
          "set": [
            "CAP_NET_RAW"
          ]
        }
      }
    ]
...

$ actool patch-manifest -capability CAP_NET_RAW,CAP_NET_BIND_SERVICE caps-retain-set-example.aci caps-retain-set-patched.aci

$ actool cat-manifest caps-retain-set-patched.aci
...
    "isolators": [
      {
        "name": "os/linux/capabilities-retain-set",
        "value": {
          "set": [
            "CAP_NET_RAW",
            "CAP_NET_BIND_SERVICE"
          ]
        }
      }
    ]
...

```

Now run the image to check that the `CAP_NET_BIND_SERVICE` capability added to
the patched image is retained as expected by using `nc` to listen on a
"privileged" port:

```
$ sudo rkt run --interactive --insecure-options=image caps-retain-set-patched.aci
image: using image from file stage1-coreos.aci
image: using image from file caps-retain-set-patched.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # nc -v -l -p 80
listening on [::]:80 ...
```

### Overriding capabilities at run-time

Capabilities can be directly overridden at run time from the command-line,
without changing the executed images.
The `--caps-retain` option to `rkt run` manipulates the `retain` capabilities set.
The `--caps-remove` option manipulates the `remove` set.

Capabilities specified from the command-line will replace all capability settings in the image manifest.
Also as stated above the options `--caps-retain`, and `--caps-remove` are mutually exclusive.
Only one can be specified at a time.

Capabilities isolators can be added on the command line at run time by
specifying the desired overriding set, as shown in this example:

```
$ sudo rkt run --interactive quay.io/coreos/alpine-sh --caps-retain CAP_NET_BIND_SERVICE
image: using image from file /usr/local/bin/stage1-coreos.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # whoami
root

/ # ping -c 1 8.8.8.8
PING 8.8.8.8 (8.8.8.8): 56 data bytes
ping: permission denied (are you root?)

```

Capability sets are application-specific configuration entries, and in a
`rkt run` command line, they must follow the application container image to
which they apply.
Each application within a pod can have different capability sets.

## Recommendations

As with most security features, capability isolators may require some
application-specific tuning in order to be maximally effective. For this reason,
for security-sensitive environments it is recommended to have a well-specified
set of capabilities requirements and follow best practices:

 1. Always follow the principle of least privilege and, whenever possible,
    avoid running applications as root
 2. Only grant the minimum set of capabilities needed by an application,
    according to its typical usage
 3. Avoid granting overly generic capabilities. For example, `CAP_SYS_ADMIN` and
    `CAP_SYS_PTRACE` are typically bad choices, as they open large attack
    surfaces.
 4. Prefer a whitelisting approach, trying to keep the "retain-set" as small as
    possible.

[acbuild]: https://github.com/containers/build
[actool]: https://github.com/appc/spec#building-acis
[capabilities]: https://lwn.net/Kernel/Index/#Capabilities
[default-caps]: https://github.com/appc/spec/blob/master/spec/ace.md#oslinuxcapabilities-remove-set
[grsec-forums]: https://forums.grsecurity.net/viewtopic.php?f=7&t=2522
[man-capabilities]: http://man7.org/linux/man-pages/man7/capabilities.7.html
