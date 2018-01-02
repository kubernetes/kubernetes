# User namespaces

## Background

User namespaces is a feature of Linux that can be used to separate the user IDs and group IDs between the host and containers.
It can provide a better isolation and security: the privileged user `root` in the container can be mapped to a non-privileged user on the host.

## Implementation status

rkt's implementation is based on systemd-nspawn.
A pod can transparently use user IDs in the range 0-65535 and this range is mapped on the host to a high range chosen randomly.

Before the pod is started, the ACIs are rendered to the filesystem and the owners of the files are set with `chown` in that high range.

## Current limitations

### UID range allocation

When starting several pods with user namespaces, they will each get a random UID range.

Although very unlikely, it is possible that two distincts containers get the same UID range.
If this happens, user namespaces will not provide any additional isolation between the two containers, exactly like when user namespaces are not used.
The two containers will however still not use the same UID range as the host, so using user namespaces is better than not using them.
In order to avoid collisions, it is planned to implement a locking mechanism so that two pods will always have a different UID range.

### Incompatible with overlayfs

The initial implementation works only with `--no-overlay`.
Ideally, preparing a pod should not have to iterate over all files to call `chown`.

It is planned to add kernel support for a mount option to shift the user IDs in the correct range (see [#1057][rkt-1057]).
It would make it work with overlayfs.

### Inconvenient UID shift on volumes

When mounting a volume from the host into the pod, the ownership of the files is not shifted, so it makes volumes difficult if not impossible to use with user namespaces.
The same kernel support should help here too ([#1057][rkt-1057]).

[rkt-1057]: https://github.com/coreos/rkt/issues/1057
