## App Environment

Apps launched by rkt have access to some basic devices and file systems as defined by the App Container spec in the [OS-SPEC](https://github.com/appc/spec/blob/master/spec/OS-SPEC.md) section.

In addition to the basic devices and file systems mandated by the App Container spec, rkt gives access to the following files.

#### /etc/hosts

Support for /etc/hosts is optional in the App Container spec. rkt creates it.

#### /etc/resolv.conf

/etc/resolv.conf is automatically prepared by rkt as described in the [DNS support](Documentation/networking/dns.md).

#### /run/systemd/journal

Since rkt v1.2.0, rkt gives access to systemd-journald's sockets in the /run/systemd/journal directory:
- /run/systemd/journal/dev-log
- /run/systemd/journal/socket
- /run/systemd/journal/stdout

#### /dev/log

Since rkt v1.2.0, if /dev/log does not exist in the image, it will be created as a symlink to /run/systemd/journal/dev-log.
