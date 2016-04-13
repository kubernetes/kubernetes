<!--[metadata]>
+++
title = "daemon"
description = "The daemon command description and usage"
keywords = ["container, daemon, runtime"]
[menu.main]
parent = "smn_cli"
+++
<![end-metadata]-->

# daemon

    Usage: docker [OPTIONS] COMMAND [arg...]

    A self-sufficient runtime for linux containers.

    Options:
      --api-cors-header=""                   Set CORS headers in the remote API
      -b, --bridge=""                        Attach containers to a network bridge
      --bip=""                               Specify network bridge IP
      --config=~/.docker                     Location of client config files
      -D, --debug=false                      Enable debug mode
      -d, --daemon=false                     Enable daemon mode
      --default-gateway=""                   Container default gateway IPv4 address
      --default-gateway-v6=""                Container default gateway IPv6 address
      --dns=[]                               DNS server to use
      --dns-search=[]                        DNS search domains to use
      --default-ulimit=[]                    Set default ulimit settings for containers
      -e, --exec-driver="native"             Exec driver to use
      --exec-opt=[]                          Set exec driver options
      --exec-root="/var/run/docker"          Root of the Docker execdriver
      --fixed-cidr=""                        IPv4 subnet for fixed IPs
      --fixed-cidr-v6=""                     IPv6 subnet for fixed IPs
      -G, --group="docker"                   Group for the unix socket
      -g, --graph="/var/lib/docker"          Root of the Docker runtime
      -H, --host=[]                          Daemon socket(s) to connect to
      -h, --help=false                       Print usage
      --icc=true                             Enable inter-container communication
      --insecure-registry=[]                 Enable insecure registry communication
      --ip=0.0.0.0                           Default IP when binding container ports
      --ip-forward=true                      Enable net.ipv4.ip_forward
      --ip-masq=true                         Enable IP masquerading
      --iptables=true                        Enable addition of iptables rules
      --ipv6=false                           Enable IPv6 networking
      -l, --log-level="info"                 Set the logging level
      --label=[]                             Set key=value labels to the daemon
      --log-driver="json-file"               Default driver for container logs
      --log-opt=[]                           Log driver specific options
      --mtu=0                                Set the containers network MTU
      -p, --pidfile="/var/run/docker.pid"    Path to use for daemon PID file
      --registry-mirror=[]                   Preferred Docker registry mirror
      -s, --storage-driver=""                Storage driver to use
      --selinux-enabled=false                Enable selinux support
      --storage-opt=[]                       Set storage driver options
      --tls=false                            Use TLS; implied by --tlsverify
      --tlscacert="~/.docker/ca.pem"         Trust certs signed only by this CA
      --tlscert="~/.docker/cert.pem"         Path to TLS certificate file
      --tlskey="~/.docker/key.pem"           Path to TLS key file
      --tlsverify=false                      Use TLS and verify the remote
      --userland-proxy=true                  Use userland proxy for loopback traffic
      -v, --version=false                    Print version information and quit

Options with [] may be specified multiple times.

The Docker daemon is the persistent process that manages containers. Docker
uses the same binary for both the daemon and client. To run the daemon you
provide the `-d` flag.

To run the daemon with debug output, use `docker -d -D`.

## Daemon socket option

The Docker daemon can listen for [Docker Remote API](/reference/api/docker_remote_api/)
requests via three different types of Socket: `unix`, `tcp`, and `fd`.

By default, a `unix` domain socket (or IPC socket) is created at
`/var/run/docker.sock`, requiring either `root` permission, or `docker` group
membership.

If you need to access the Docker daemon remotely, you need to enable the `tcp`
Socket. Beware that the default setup provides un-encrypted and
un-authenticated direct access to the Docker daemon - and should be secured
either using the [built in HTTPS encrypted socket](/articles/https/), or by
putting a secure web proxy in front of it. You can listen on port `2375` on all
network interfaces with `-H tcp://0.0.0.0:2375`, or on a particular network
interface using its IP address: `-H tcp://192.168.59.103:2375`. It is
conventional to use port `2375` for un-encrypted, and port `2376` for encrypted
communication with the daemon.

> **Note:**
> If you're using an HTTPS encrypted socket, keep in mind that only
> TLS1.0 and greater are supported. Protocols SSLv3 and under are not
> supported anymore for security reasons.

On Systemd based systems, you can communicate with the daemon via
[Systemd socket activation](http://0pointer.de/blog/projects/socket-activation.html),
use `docker -d -H fd://`. Using `fd://` will work perfectly for most setups but
you can also specify individual sockets: `docker -d -H fd://3`. If the
specified socket activated files aren't found, then Docker will exit. You can
find examples of using Systemd socket activation with Docker and Systemd in the
[Docker source tree](https://github.com/docker/docker/tree/master/contrib/init/systemd/).

You can configure the Docker daemon to listen to multiple sockets at the same
time using multiple `-H` options:

    # listen using the default unix socket, and on 2 specific IP addresses on this host.
    docker -d -H unix:///var/run/docker.sock -H tcp://192.168.59.106 -H tcp://10.10.10.2

The Docker client will honor the `DOCKER_HOST` environment variable to set the
`-H` flag for the client.

    $ docker -H tcp://0.0.0.0:2375 ps
    # or
    $ export DOCKER_HOST="tcp://0.0.0.0:2375"
    $ docker ps
    # both are equal

Setting the `DOCKER_TLS_VERIFY` environment variable to any value other than
the empty string is equivalent to setting the `--tlsverify` flag. The following
are equivalent:

    $ docker --tlsverify ps
    # or
    $ export DOCKER_TLS_VERIFY=1
    $ docker ps

The Docker client will honor the `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`
environment variables (or the lowercase versions thereof). `HTTPS_PROXY` takes
precedence over `HTTP_PROXY`.

### Daemon storage-driver option

The Docker daemon has support for several different image layer storage
drivers: `aufs`, `devicemapper`, `btrfs`, `zfs` and `overlay`.

The `aufs` driver is the oldest, but is based on a Linux kernel patch-set that
is unlikely to be merged into the main kernel. These are also known to cause
some serious kernel crashes. However, `aufs` is also the only storage driver
that allows containers to share executable and shared library memory, so is a
useful choice when running thousands of containers with the same program or
libraries.

The `devicemapper` driver uses thin provisioning and Copy on Write (CoW)
snapshots. For each devicemapper graph location – typically
`/var/lib/docker/devicemapper` – a thin pool is created based on two block
devices, one for data and one for metadata. By default, these block devices
are created automatically by using loopback mounts of automatically created
sparse files. Refer to [Storage driver options](#storage-driver-options) below
for a way how to customize this setup.
[~jpetazzo/Resizing Docker containers with the Device Mapper plugin](http://jpetazzo.github.io/2014/01/29/docker-device-mapper-resize/)
article explains how to tune your existing setup without the use of options.

The `btrfs` driver is very fast for `docker build` - but like `devicemapper`
does not share executable memory between devices. Use
`docker -d -s btrfs -g /mnt/btrfs_partition`.

The `zfs` driver is probably not fast as `btrfs` but has a longer track record
on stability. Thanks to `Single Copy ARC` shared blocks between clones will be
cached only once. Use `docker -d -s zfs`. To select a different zfs filesystem
set `zfs.fsname` option as described in [Storage driver options](#storage-driver-options).

The `overlay` is a very fast union filesystem. It is now merged in the main
Linux kernel as of [3.18.0](https://lkml.org/lkml/2014/10/26/137). Call
`docker -d -s overlay` to use it.

> **Note:**
> As promising as `overlay` is, the feature is still quite young and should not
> be used in production. Most notably, using `overlay` can cause excessive
> inode consumption (especially as the number of images grows), as well as
> being incompatible with the use of RPMs.

> **Note:**
> It is currently unsupported on `btrfs` or any Copy on Write filesystem
> and should only be used over `ext4` partitions.

### Storage driver options

Particular storage-driver can be configured with options specified with
`--storage-opt` flags. Options for `devicemapper` are prefixed with `dm` and
options for `zfs` start with `zfs`.

*  `dm.thinpooldev`

     Specifies a custom block storage device to use for the thin pool.

     If using a block device for device mapper storage, it is best to use `lvm`
     to create and manage the thin-pool volume. This volume is then handed to Docker
     to exclusively create snapshot volumes needed for images and containers.  

     Managing the thin-pool outside of Docker makes for the most feature-rich
     method of having Docker utilize device mapper thin provisioning as the
     backing storage for Docker's containers. The highlights of the lvm-based
     thin-pool management feature include: automatic or interactive thin-pool
     resize support, dynamically changing thin-pool features, automatic thinp
     metadata checking when lvm activates the thin-pool, etc.

     Example use:

        docker -d --storage-opt dm.thinpooldev=/dev/mapper/thin-pool

 *  `dm.basesize`

    Specifies the size to use when creating the base device, which limits the
    size of images and containers. The default value is 10G. Note, thin devices
    are inherently "sparse", so a 10G device which is mostly empty doesn't use
    10 GB of space on the pool. However, the filesystem will use more space for
    the empty case the larger the device is.

    This value affects the system-wide "base" empty filesystem
    that may already be initialized and inherited by pulled images. Typically,
    a change to this value requires additional steps to take effect:

        $ sudo service docker stop
        $ sudo rm -rf /var/lib/docker
        $ sudo service docker start

    Example use:

        $ docker -d --storage-opt dm.basesize=20G

 *  `dm.loopdatasize`

    >**Note**: This option configures devicemapper loopback, which should not be used in production.
		
    Specifies the size to use when creating the loopback file for the
    "data" device which is used for the thin pool. The default size is
    100G. The file is sparse, so it will not initially take up this
    much space.

    Example use:

        $ docker -d --storage-opt dm.loopdatasize=200G

 *  `dm.loopmetadatasize`

    >**Note**: This option configures devicemapper loopback, which should not be used in production.

    Specifies the size to use when creating the loopback file for the
    "metadadata" device which is used for the thin pool. The default size
    is 2G. The file is sparse, so it will not initially take up
    this much space.

    Example use:

        $ docker -d --storage-opt dm.loopmetadatasize=4G

 *  `dm.fs`

    Specifies the filesystem type to use for the base device. The supported
    options are "ext4" and "xfs". The default is "ext4"

    Example use:

        $ docker -d --storage-opt dm.fs=xfs

 *  `dm.mkfsarg`

    Specifies extra mkfs arguments to be used when creating the base device.

    Example use:

        $ docker -d --storage-opt "dm.mkfsarg=-O ^has_journal"

 *  `dm.mountopt`

    Specifies extra mount options used when mounting the thin devices.

    Example use:

        $ docker -d --storage-opt dm.mountopt=nodiscard

 *  `dm.datadev`

    (Deprecated, use `dm.thinpooldev`)

    Specifies a custom blockdevice to use for data for the thin pool.

    If using a block device for device mapper storage, ideally both datadev and
    metadatadev should be specified to completely avoid using the loopback
    device.

    Example use:

        $ docker -d --storage-opt dm.datadev=/dev/sdb1 --storage-opt dm.metadatadev=/dev/sdc1

 *  `dm.metadatadev`

    (Deprecated, use `dm.thinpooldev`)

    Specifies a custom blockdevice to use for metadata for the thin pool.

    For best performance the metadata should be on a different spindle than the
    data, or even better on an SSD.

    If setting up a new metadata pool it is required to be valid. This can be
    achieved by zeroing the first 4k to indicate empty metadata, like this:

	$ dd if=/dev/zero of=$metadata_dev bs=4096 count=1

    Example use:

        $ docker -d --storage-opt dm.datadev=/dev/sdb1 --storage-opt dm.metadatadev=/dev/sdc1

 *  `dm.blocksize`

    Specifies a custom blocksize to use for the thin pool. The default
    blocksize is 64K.

    Example use:

        $ docker -d --storage-opt dm.blocksize=512K

 *  `dm.blkdiscard`

    Enables or disables the use of blkdiscard when removing devicemapper
    devices. This is enabled by default (only) if using loopback devices and is
    required to resparsify the loopback file on image/container removal.

    Disabling this on loopback can lead to *much* faster container removal
    times, but will make the space used in `/var/lib/docker` directory not be
    returned to the system for other use when containers are removed.

    Example use:

        $ docker -d --storage-opt dm.blkdiscard=false

 *  `dm.override_udev_sync_check`

    Overrides the `udev` synchronization checks between `devicemapper` and `udev`.
    `udev` is the device manager for the Linux kernel.

    To view the `udev` sync support of a Docker daemon that is using the
    `devicemapper` driver, run:

        $ docker info
        [...]
        Udev Sync Supported: true
        [...]

    When `udev` sync support is `true`, then `devicemapper` and udev can
    coordinate the activation and deactivation of devices for containers.

    When `udev` sync support is `false`, a race condition occurs between
    the`devicemapper` and `udev` during create and cleanup. The race condition
    results in errors and failures. (For information on these failures, see
    [docker#4036](https://github.com/docker/docker/issues/4036))

    To allow the `docker` daemon to start, regardless of `udev` sync not being
    supported, set `dm.override_udev_sync_check` to true:

        $ docker -d --storage-opt dm.override_udev_sync_check=true

    When this value is `true`, the  `devicemapper` continues and simply warns
    you the errors are happening.

    > **Note:**
    > The ideal is to pursue a `docker` daemon and environment that does
    > support synchronizing with `udev`. For further discussion on this
    > topic, see [docker#4036](https://github.com/docker/docker/issues/4036).
    > Otherwise, set this flag for migrating existing Docker daemons to
    > a daemon with a supported environment.


## Docker execdriver option

Currently supported options of `zfs`:

 * `zfs.fsname`

    Set zfs filesystem under which docker will create its own datasets.
    By default docker will pick up the zfs filesystem where docker graph
    (`/var/lib/docker`) is located.

    Example use:

        $ docker -d -s zfs --storage-opt zfs.fsname=zroot/docker

## Docker execdriver option

The Docker daemon uses a specifically built `libcontainer` execution driver as
its interface to the Linux kernel `namespaces`, `cgroups`, and `SELinux`.

There is still legacy support for the original [LXC userspace tools](
https://linuxcontainers.org/) via the `lxc` execution driver, however, this is
not where the primary development of new functionality is taking place.
Add `-e lxc` to the daemon flags to use the `lxc` execution driver.

## Options for the native execdriver

You can configure the `native` (libcontainer) execdriver using options specified
with the `--exec-opt` flag. All the flag's options have the `native` prefix. A
single `native.cgroupdriver` option is available.

The `native.cgroupdriver` option specifies the management of the container's
cgroups. You can specify `cgroupfs` or `systemd`. If you specify `systemd` and
it is not available, the system uses `cgroupfs`. By default, if no option is
specified, the execdriver first tries `systemd` and falls back to `cgroupfs`.
This example sets the execdriver to `cgroupfs`:

    $ sudo docker -d --exec-opt native.cgroupdriver=cgroupfs

Setting this option applies to all containers the daemon launches.

## Daemon DNS options

To set the DNS server for all Docker containers, use
`docker -d --dns 8.8.8.8`.

To set the DNS search domain for all Docker containers, use
`docker -d --dns-search example.com`.

## Insecure registries

Docker considers a private registry either secure or insecure. In the rest of
this section, *registry* is used for *private registry*, and `myregistry:5000`
is a placeholder example for a private registry.

A secure registry uses TLS and a copy of its CA certificate is placed on the
Docker host at `/etc/docker/certs.d/myregistry:5000/ca.crt`. An insecure
registry is either not using TLS (i.e., listening on plain text HTTP), or is
using TLS with a CA certificate not known by the Docker daemon. The latter can
happen when the certificate was not found under
`/etc/docker/certs.d/myregistry:5000/`, or if the certificate verification
failed (i.e., wrong CA).

By default, Docker assumes all, but local (see local registries below),
registries are secure. Communicating with an insecure registry is not possible
if Docker assumes that registry is secure. In order to communicate with an
insecure registry, the Docker daemon requires `--insecure-registry` in one of
the following two forms:

* `--insecure-registry myregistry:5000` tells the Docker daemon that
  myregistry:5000 should be considered insecure.
* `--insecure-registry 10.1.0.0/16` tells the Docker daemon that all registries
  whose domain resolve to an IP address is part of the subnet described by the
  CIDR syntax, should be considered insecure.

The flag can be used multiple times to allow multiple registries to be marked
as insecure.

If an insecure registry is not marked as insecure, `docker pull`,
`docker push`, and `docker search` will result in an error message prompting
the user to either secure or pass the `--insecure-registry` flag to the Docker
daemon as described above.

Local registries, whose IP address falls in the 127.0.0.0/8 range, are
automatically marked as insecure as of Docker 1.3.2. It is not recommended to
rely on this, as it may change in the future.

## Running a Docker daemon behind a HTTPS_PROXY

When running inside a LAN that uses a `HTTPS` proxy, the Docker Hub
certificates will be replaced by the proxy's certificates. These certificates
need to be added to your Docker host's configuration:

1. Install the `ca-certificates` package for your distribution
2. Ask your network admin for the proxy's CA certificate and append them to
   `/etc/pki/tls/certs/ca-bundle.crt`
3. Then start your Docker daemon with `HTTPS_PROXY=http://username:password@proxy:port/ docker -d`.
   The `username:` and `password@` are optional - and are only needed if your
   proxy is set up to require authentication.

This will only add the proxy and authentication to the Docker daemon's requests -
your `docker build`s and running containers will need extra configuration to
use the proxy

## Default Ulimits

`--default-ulimit` allows you to set the default `ulimit` options to use for
all containers. It takes the same options as `--ulimit` for `docker run`. If
these defaults are not set, `ulimit` settings will be inherited, if not set on
`docker run`, from the Docker daemon. Any `--ulimit` options passed to 
`docker run` will overwrite these defaults.

Be careful setting `nproc` with the `ulimit` flag as `nproc` is designed by Linux to
set the maximum number of processes available to a user, not to a container. For details
please check the [run](run.md) reference.

## Miscellaneous options

IP masquerading uses address translation to allow containers without a public
IP to talk to other machines on the Internet. This may interfere with some
network topologies and can be disabled with --ip-masq=false.

Docker supports softlinks for the Docker data directory (`/var/lib/docker`) and
for `/var/lib/docker/tmp`. The `DOCKER_TMPDIR` and the data directory can be
set like this:

    DOCKER_TMPDIR=/mnt/disk2/tmp /usr/local/bin/docker -d -D -g /var/lib/docker -H unix:// > /var/lib/boot2docker/docker.log 2>&1
    # or
    export DOCKER_TMPDIR=/mnt/disk2/tmp
    /usr/local/bin/docker -d -D -g /var/lib/docker -H unix:// > /var/lib/boot2docker/docker.log 2>&1


