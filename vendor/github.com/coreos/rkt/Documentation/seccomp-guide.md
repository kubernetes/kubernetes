# Seccomp Isolators Guide

This document is a walk-through guide describing how to use rkt isolators for
[Linux seccomp filtering][lwn-seccomp].

* [About Seccomp](#about-seccomp)
* [Predefined Seccomp Filters](#predefined-seccomp-filter)
* [Seccomp Isolators](#seccomp-isolators)
* [Usage Example](#usage-example)
* [Overriding Seccomp Filters](#overriding-seccomp-filters)
* [Recommendations](#recommendations)

## About seccomp

Linux seccomp (short for SECure COMputing) filtering allows one to specify which
system calls a process should be allowed to invoke, reducing the kernel surface
exposed to applications.
This provides a clearly defined mechanism to build sandboxed environments, where
processes can run having access only to a specific reduced set of system calls.

In the context of containers, seccomp filtering is useful for:

* Restricting applications from invoking syscalls that can affect the host
* Reducing kernel attack surface in case of security bugs

For more details on how Linux seccomp filtering works, see
[seccomp(2)][man-seccomp].

## Predefined seccomp filters

By default, rkt comes with a set of predefined filtering groups that can be
used to quickly build sandboxed environments for containerized applications.
Each set is simply a reference to a group of syscalls, covering a single
functional area or kernel subsystem. They can be further combined to
build more complex filters, either by blacklisting or by whitelisting specific
system calls. To distinguish these predefined groups from real syscall names,
wildcard labels are prefixed with a `@` symbols and are namespaced.

The App Container Spec (appc) defines
[two groups][appc-isolators]:

 * `@appc.io/all` represents the set of all available syscalls.
 * `@appc.io/empty` represents the empty set.

rkt provides two default groups for generic usage:

 * `@rkt/default-blacklist` represents a broad-scope filter than can be used for generic blacklisting
 * `@rkt/default-whitelist` represents a broad-scope filter than can be used for generic whitelisting

For compatibility reasons, two groups are provided mirroring [default Docker profiles][docker-seccomp]:

 * `@docker/default-blacklist`
 * `@docker/default-whitelist`

When using stage1 images with systemd >= v231, some
[predefined groups][systemd-seccomp]
are also available:

 * `@systemd/clock` for syscalls manipulating the system clock
 * `@systemd/default-whitelist` for a generic set of typically whitelisted syscalls
 * `@systemd/mount` for filesystem mounting and unmounting
 * `@systemd/network-io` for socket I/O operationgs
 * `@systemd/obsolete` for unusual, obsolete or unimplemented syscalls
 * `@systemd/privileged` for syscalls which need super-user syscalls
 * `@systemd/process` for syscalls acting on process control, execution and namespacing
 * `@systemd/raw-io` for raw I/O port access

When no seccomp filtering is specified, by default rkt whitelists all the generic
syscalls typically needed by applications for common operations. This is
the same set defined by `@rkt/default-whitelist`.

The default set is tailored to stop applications from performing a large
variety of privileged actions, while not impacting their normal behavior.
Operations which are typically not needed in containers and which may
impact host state, eg. invoking [`umount(2)`][man-umount], are denied in this way.

However, this default set is mostly meant as a safety precaution against erratic
and misbehaving applications, and will not suffice against tailored attacks.
As such, it is recommended to fine-tune seccomp filtering using one of the
customizable isolators available in rkt.

## Seccomp Isolators

When running Linux containers, rkt provides two mutually exclusive isolators
to define a seccomp filter for an application:

* `os/linux/seccomp-retain-set`
* `os/linux/seccomp-remove-set`

Those isolators cover different use-cases and employ different techniques to
achieve the same goal of limiting available syscalls. As such, they cannot
be used together at the same time, and recommended usage varies on a
case-by-case basis.

### Operation mode

Seccomp isolators work by defining a set of syscalls than can be either blocked
("remove-set") or allowed ("retain-set"). Once an application tries to invoke
a blocked syscall, the kernel will deny this operation and the application will
be notified about the failure.

By default, invoking blocked syscalls will result in the application being
immediately terminated with a `SIGSYS` signal. This behavior can be tweaked by
returning a specific error code ("errno") to the application instead of
terminating it.

For both isolators, this can be customized by specifying an additional `errno`
parameter with the desired symbolic errno name. For a list of errno labels, check
the [reference][man-errno] at `man 3 errno`.

### Retain-set

`os/linux/seccomp-retain-set` allows for an additive approach to build a seccomp
filter: applications will not able to use any syscalls, except the ones
listed in this isolator.

This whitelisting approach is useful for completely locking down environments
and whenever application requirements (in terms of syscalls) are
well-defined in advance. It allows one to ensure that exactly and only the
specified syscalls could ever be used.

For example, the "retain-set" for a typical network application will include
entries for generic POSIX operations (available in `@systemd/default-whitelist`),
socket operations (`@systemd/network-io`) and reacting to I/O
events (`@systemd/io-event`).

### Remove-set

`os/linux/seccomp-remove-set` tackles syscalls in a subtractive way:
starting from all available syscalls, single entries can be forbidden in order
to prevent specific actions.

This blacklisting approach is useful to somehow limit applications which have
broad requirements in terms of syscalls, in order to deny access to some clearly
unused but potentially exploitable syscalls.

For example, an application that will need to perform multiple operations but is
known to never touch mountpoints could have `@systemd/mount` specified in its
"remove-set".

## Usage Example

The goal of these examples is to show how to build ACI images with [`acbuild`][acbuild],
where some syscalls are either explicitly blocked or allowed.
For simplicity, the starting point will be a bare Alpine Linux image which
ships with `ping` and `umount` commands (from busybox). Those
commands respectively requires [`socket(2)`][man-socket] and [`umount(2)`][man-umount] syscalls in order to
perform privileged operations.
To block their usage, a syscalls filter can be installed via
`os/linux/seccomp-remove-set` or `os/linux/seccomp-retain-set`; both approaches
are shown here.

### Blacklisting specific syscalls

This example shows how to block socket operation (e.g. with `ping`), by removing
`socket()` from the set of allowed syscalls.

First, a local image is built with an explicit "remove-set" isolator.
This set contains the syscalls that need to be forbidden in order to block
socket setup:

```
$ acbuild begin
$ acbuild set-name localhost/seccomp-remove-set-example
$ acbuild dependency add quay.io/coreos/alpine-sh
$ acbuild set-exec -- /bin/sh
$ echo '{ "set": ["@rkt/default-blacklist", "socket"] }' | acbuild isolator add "os/linux/seccomp-remove-set" -
$ acbuild write seccomp-remove-set-example.aci
$ acbuild end
```

Once properly built, this image can be run in order to check that `ping` usage is
now blocked by the seccomp filter. At the same time, the default blacklist will
also block other dangerous syscalls like `umount(2)`:

```
$ sudo rkt run --interactive --insecure-options=image seccomp-remove-set-example.aci
image: using image from file stage1-coreos.aci
image: using image from file seccomp-remove-set-example.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # whoami
root

/ # ping -c1 8.8.8.8
PING 8.8.8.8 (8.8.8.8): 56 data bytes
Bad system call

/ # umount /proc/bus/
Bad system call
```

This means that `socket(2)` and `umount(2)` have been both effectively disabled
inside the container.

### Allowing specific syscalls

In contrast to the example above, this one shows how to allow some operations
only (e.g. network communication via `ping`), by whitelisting all required
syscalls. This means that syscalls outside of this set will be blocked.

First, a local image is built with an explicit "retain-set" isolator.
This set contains the rkt wildcard "default-whitelist" (which already provides
all socket-related entries), plus some custom syscalls (e.g. `umount(2)`) which
are typically not allowed:

```
$ acbuild begin
$ acbuild set-name localhost/seccomp-retain-set-example
$ acbuild dependency add quay.io/coreos/alpine-sh
$ acbuild set-exec -- /bin/sh
$ echo '{ "set": ["@rkt/default-whitelist", "umount", "umount2"] }' | acbuild isolator add "os/linux/seccomp-retain-set" -
$ acbuild write seccomp-retain-set-example.aci
$ acbuild end
```

Once run, it can be easily verified that both `ping` and `umount` are now
functional inside the container. These operations also require [additional
capabilities][capabilities-guide] to be retained in order to work:

```
$ sudo rkt run --interactive --insecure-options=image seccomp-retain-set-example.aci --caps-retain=CAP_SYS_ADMIN,CAP_NET_RAW
image: using image from file stage1-coreos.aci
image: using image from file seccomp-retain-set-example.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # whoami
root

/ # ping -c 1 8.8.8.8
PING 8.8.8.8 (8.8.8.8): 56 data bytes
64 bytes from 8.8.8.8: seq=0 ttl=41 time=24.910 ms

--- 8.8.8.8 ping statistics ---
1 packets transmitted, 1 packets received, 0% packet loss
round-trip min/avg/max = 24.910/24.910/24.910 ms

/ # mount | grep /proc/bus
proc on /proc/bus type proc (ro,nosuid,nodev,noexec,relatime)
/ # umount /proc/bus
/ # mount | grep /proc/bus
```

However, others syscalls are still not available to the application.
For example, trying to set the time will result in a failure due to invoking
non-whitelisted syscalls:

```
$ sudo rkt run --interactive --insecure-options=image seccomp-retain-set-example.aci
image: using image from file stage1-coreos.aci
image: using image from file seccomp-retain-set-example.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # whoami
root

/ # adjtimex -f 0
Bad system call
```

## Overriding Seccomp Filters

Seccomp filters are typically defined when creating images, as they are tightly
linked to specific app requirements. However, image consumers may need to further
tweak/restrict the set of available syscalls in specific local scenarios.
This can be done either by permanently patching the manifest of specific images,
or by overriding seccomp isolators with command line options.

### Patching images

Image manifests can be manipulated manually, by unpacking the image and editing
the manifest file, or with helper tools like [`actool`][actool].
To override an image's pre-defined syscalls set, just replace the existing seccomp
isolators in the image with new isolators defining the desired syscalls.

The `patch-manifest` subcommand to `actool` manipulates the syscalls sets
defined in an image.
`actool patch-manifest -seccomp-mode=... -seccomp-set=...` options
can be used together to override any seccomp filters by specifying a new mode
(retain or reset), an optional custom errno, and a set of syscalls to filter.
These commands take an input image, modify any existing seccomp isolators, and
write the changes to an output image, as shown in the example:

```
$ actool cat-manifest seccomp-retain-set-example.aci
...
    "isolators": [
      {
      "name": "os/linux/seccomp-retain-set",
        "value": {
          "set": [
            "@rkt/default-whitelist",
            "umount",
            "umount2"
          ]
        }
      }
    ]
...

$ actool patch-manifest -seccomp-mode=retain,errno=ENOSYS -seccomp-set=@rkt/default-whitelist seccomp-retain-set-example.aci seccomp-retain-set-patched.aci

$ actool cat-manifest seccomp-retain-set-patched.aci
...
    "isolators": [
      {
        "name": "os/linux/seccomp-retain-set",
        "value": {
          "set": [
            "@rkt/default-whitelist",
          ],
          "errno": "ENOSYS"
        }
      }
    ]
...

```

Now run the image to verify that the `umount(2)` syscall is no longer allowed,
and a custom error is returned:

```
$ sudo rkt run --interactive --insecure-options=image seccomp-retain-set-patched.aci
image: using image from file stage1-coreos.aci
image: using image from file seccomp-retain-set-patched.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # mount | grep /proc/bus
proc on /proc/bus type proc (ro,nosuid,nodev,noexec,relatime)
/ # umount /proc/bus/
umount: can't umount /proc/bus: Function not implemented
```

### Overriding seccomp filters at run-time

Seccomp filters can be directly overridden at run time from the command-line,
without changing the executed images.
The `--seccomp` option to `rkt run` can manipulate both the "retain" and the
"remove" isolators.

Isolator overridden from the command-line will replace all seccomp settings in
the image manifest, and can be specified as shown in this example:

```
$ sudo rkt run --interactive quay.io/coreos/alpine-sh --seccomp mode=remove,errno=ENOTSUP,socket
image: using image from file /usr/local/bin/stage1-coreos.aci
image: using image from local store for image name quay.io/coreos/alpine-sh

/ # whoami
root

/ # ping -c 1 8.8.8.8
PING 8.8.8.8 (8.8.8.8): 56 data bytes
ping: can't create raw socket: Not supported
```

Seccomp isolators are application-specific configuration entries, and in a
`rkt run` command line they **must follow the application container image to
which they apply**.
Each application within a pod can have different seccomp filters.

## Recommendations

As with most security features, seccomp isolators may require some
application-specific tuning in order to be maximally effective. For this reason,
for security-sensitive environments it is recommended to have a well-specified
set of syscalls requirements and follow best practices:

 1. Only allow syscalls needed by an application, according to its typical usage.
 2. While it is possible to completely disable seccomp, it is rarely needed and
    should be generally avoided. Tweaking the syscalls set is a better approach
    instead.
 3. Avoid granting access to dangerous syscalls. For example, [`mount(2)`][man-mount] and
    [`ptrace(2)`][man-ptrace] are typically abused to escape containers.
 4. Prefer a whitelisting approach, trying to keep the "retain-set" as small as
    possible.


[acbuild]: https://github.com/containers/build
[actool]: https://github.com/appc/spec#building-acis
[appc-isolators]: https://github.com/appc/spec/blob/master/spec/ace.md#linux-isolators
[capabilities-guide]: capabilities-guide.md
[docker-seccomp]: https://docs.docker.com/engine/security/seccomp/
[lwn-seccomp]: https://lwn.net/Articles/656307/
[man-errno]: http://man7.org/linux/man-pages/man3/errno.3.html
[man-mount]: http://man7.org/linux/man-pages/man2/mount.2.html
[man-ptrace]: http://man7.org/linux/man-pages/man2/ptrace.2.html
[man-seccomp]: http://man7.org/linux/man-pages/man2/seccomp.2.html
[man-socket]: http://man7.org/linux/man-pages/man2/socket.2.html
[man-umount]: http://man7.org/linux/man-pages/man2/umount.2.html
[systemd-seccomp]: https://www.freedesktop.org/software/systemd/man/systemd.exec.html#SystemCallFilter=
