<!--[metadata]>
+++
title = "Installation on Oracle Linux"
description = "Installation instructions for Docker on Oracle Linux."
keywords = ["Docker, Docker documentation, requirements, linux, rhel, centos, oracle,  ol"]
[menu.main]
parent = "smn_linux"
+++
<![end-metadata]-->

# Oracle Linux 6 and 7

You do not require an Oracle Linux Support subscription to install Docker on
Oracle Linux.

*For Oracle Linux customers with an active support subscription:*
Docker is available in either the `ol6_x86_64_addons` or `ol7_x86_64_addons`
channel for Oracle Linux 6 and Oracle Linux 7 on the [Unbreakable Linux Network
(ULN)](https://linux.oracle.com).

*For Oracle Linux users without an active support subscription:*
Docker is available in the appropriate `ol6_addons` or `ol7_addons` repository
on [Oracle Public Yum](http://public-yum.oracle.com).

Docker requires the use of the Unbreakable Enterprise Kernel Release 3 (3.8.13)
or higher on Oracle Linux. This kernel supports the Docker btrfs storage engine
on both Oracle Linux 6 and 7.

Due to current Docker limitations, Docker is only able to run only on the x86_64
architecture.

## To enable the *addons* channel via the Unbreakable Linux Network:

1. Enable either the *ol6\_x86\_64\_addons* or *ol7\_x86\_64\_addons* channel
via the ULN web interface.
Consult the [Unbreakable Linux Network User's
Guide](http://docs.oracle.com/cd/E52668_01/E39381/html/index.html) for
documentation on subscribing to channels.

## To enable the *addons* repository via Oracle Public Yum:

The latest release of Oracle Linux 6 and 7 are automatically configured to use
the Oracle Public Yum repositories during installation. However, the *addons*
repository is not enabled by default.

To enable the *addons* repository:

1. Edit either `/etc/yum.repos.d/public-yum-ol6.repo` or
`/etc/yum.repos.d/public-yum-ol7.repo`
and set `enabled=1` in the `[ol6_addons]` or the `[ol7_addons]` stanza.

## Installation 

1. Ensure the appropriate *addons* channel or repository has been enabled.

2. Use yum to install the Docker package:

        $ sudo yum install docker

## Starting Docker 

1. Now that it's installed, start the Docker daemon:

    1. On Oracle Linux 6:

            $ sudo service docker start

    2. On Oracle Linux 7:

            $ sudo systemctl start docker.service

2. If you want the Docker daemon to start automatically at boot:

    1. On Oracle Linux 6:

            $ sudo chkconfig docker on

    2. On Oracle Linux 7:

            $ sudo systemctl enable docker.service

**Done!**

## Custom daemon options

If you need to add an HTTP Proxy, set a different directory or partition for the
Docker runtime files, or make other customizations, read our systemd article to
learn how to [customize your systemd Docker daemon options](/articles/systemd/).

## Using the btrfs storage engine

Docker on Oracle Linux 6 and 7 supports the use of the btrfs storage engine.
Before enabling btrfs support, ensure that `/var/lib/docker` is stored on a
btrfs-based filesystem. Review [Chapter
5](http://docs.oracle.com/cd/E37670_01/E37355/html/ol_btrfs.html) of the [Oracle
Linux Administrator's Solution
Guide](http://docs.oracle.com/cd/E37670_01/E37355/html/index.html) for details
on how to create and mount btrfs filesystems.

To enable btrfs support on Oracle Linux:

1. Ensure that `/var/lib/docker` is on a btrfs filesystem.
1. Edit `/etc/sysconfig/docker` and add `-s btrfs` to the `OTHER_ARGS` field.
2. Restart the Docker daemon:

You can now continue with the [Docker User Guide](/userguide/).

## Uninstallation

To uninstall the Docker package:

    $ sudo yum -y remove docker

The above command will not remove images, containers, volumes, or user created
configuration files on your host. If you wish to delete all images, containers,
and volumes run the following command:

    $ rm -rf /var/lib/docker

You must delete the user created configuration files manually.

## Known issues

### Docker unmounts btrfs filesystem on shutdown
If you're running Docker using the btrfs storage engine and you stop the Docker
service, it will unmount the btrfs filesystem during the shutdown process. You
should ensure the filesystem is mounted properly prior to restarting the Docker
service.

On Oracle Linux 7, you can use a `systemd.mount` definition and modify the
Docker `systemd.service` to depend on the btrfs mount defined in systemd.

### SElinux support on Oracle Linux 7
SElinux must be set to `Permissive` or `Disabled` in `/etc/sysconfig/selinux` to
use the btrfs storage engine on Oracle Linux 7.

## Further issues?

If you have a current Basic or Premier Support Subscription for Oracle Linux,
you can report any issues you have with the installation of Docker via a Service
Request at [My Oracle Support](http://support.oracle.com).

If you do not have an Oracle Linux Support Subscription, you can use the [Oracle
Linux
Forum](https://community.oracle.com/community/server_%26_storage_systems/linux/oracle_linux) for community-based support.
