<!--[metadata]>
+++
title = "Installation on Debian"
description = "Instructions for installing Docker on Debian."
keywords = ["Docker, Docker documentation, installation,  debian"]
[menu.main]
parent = "smn_linux"
+++
<![end-metadata]-->

# Debian

Docker is supported on the following versions of Debian:

 - [*Debian 8.0 Jessie (64-bit)*](#debian-jessie-80-64-bit)
 - [*Debian 7.7 Wheezy (64-bit)*](#debian-wheezy-stable-7-x-64-bit)

## Debian Jessie 8.0 (64-bit)

Debian 8 comes with a 3.16.0 Linux kernel, the `docker.io` package can be found in the `jessie-backports` repository. Reasoning behind this can be found <a href="https://lists.debian.org/debian-release/2015/03/msg00685.html" target="_blank">here</a>. Instructions how to enable the backports repository can be found <a href="http://backports.debian.org/Instructions/" target="_blank">here</a>.

> **Note**:
> Debian contains a much older KDE3/GNOME2 package called ``docker``, so the
> package and the executable are called ``docker.io``.

### Installation

Make sure you enabled the `jessie-backports` repository, as stated above.

To install the latest Debian package (may not be the latest Docker release):

    $ sudo apt-get update
    $ sudo apt-get install docker.io

To verify that everything has worked as expected:

    $ sudo docker run --rm hello-world

This command downloads and runs the `hello-world` image in a container. When the
container runs, it prints an informational message. Then, it exits.

> **Note**:
> If you want to enable memory and swap accounting see
> [this](/installation/ubuntulinux/#memory-and-swap-accounting).

### Uninstallation

To uninstall the Docker package:

    $ sudo apt-get purge docker-io

To uninstall the Docker package and dependencies that are no longer needed:

    $ sudo apt-get autoremove --purge docker-io

The above commands will not remove images, containers, volumes, or user created
configuration files on your host. If you wish to delete all images, containers,
and volumes run the following command:

    $ rm -rf /var/lib/docker

You must delete the user created configuration files manually.

## Debian Wheezy/Stable 7.x (64-bit)

Docker requires Kernel 3.8+, while Wheezy ships with Kernel 3.2 (for more details
on why 3.8 is required, see discussion on
[bug #407](https://github.com/docker/docker/issues/407)).

Fortunately, wheezy-backports currently has [Kernel 3.16
](https://packages.debian.org/search?suite=wheezy-backports&section=all&arch=any&searchon=names&keywords=linux-image-amd64),
which is officially supported by Docker.

### Installation

1. Install Kernel from wheezy-backports

    Add the following line to your `/etc/apt/sources.list`

    `deb http://http.debian.net/debian wheezy-backports main`

    then install the `linux-image-amd64` package (note the use of
    `-t wheezy-backports`)

        $ sudo apt-get update
        $ sudo apt-get install -t wheezy-backports linux-image-amd64

2. Restart your system. This is necessary for Debian to use your new kernel.

3. Install Docker using the get.docker.com script:

    `curl -sSL https://get.docker.com/ | sh`

>**Note**: If your company is behind a filtering proxy, you may find that the
>`apt-key`
>command fails for the Docker repo during installation. To work around this,
>add the key directly using the following:
>
>       $ wget -qO- https://get.docker.com/gpg | sudo apt-key add -

### Uninstallation

To uninstall the Docker package:

    $ sudo apt-get purge lxc-docker

To uninstall the Docker package and dependencies that are no longer needed:

    $ sudo apt-get autoremove --purge lxc-docker

The above commands will not remove images, containers, volumes, or user created
configuration files on your host. If you wish to delete all images, containers,
and volumes run the following command:

    $ rm -rf /var/lib/docker

You must delete the user created configuration files manually.

## Giving non-root access

The `docker` daemon always runs as the `root` user and the `docker`
daemon binds to a Unix socket instead of a TCP port. By default that
Unix socket is owned by the user `root`, and so, by default, you can
access it with `sudo`.

If you (or your Docker installer) create a Unix group called `docker`
and add users to it, then the `docker` daemon will make the ownership of
the Unix socket read/writable by the `docker` group when the daemon
starts. The `docker` daemon must always run as the root user, but if you
run the `docker` client as a user in the `docker` group then you don't
need to add `sudo` to all the client commands. From Docker 0.9.0 you can
use the `-G` flag to specify an alternative group.

> **Warning**:
> The `docker` group (or the group specified with the `-G` flag) is
> `root`-equivalent; see [*Docker Daemon Attack Surface*](
> /articles/security/#docker-daemon-attack-surface) details.

**Example:**

    # Add the docker group if it doesn't already exist.
    $ sudo groupadd docker

    # Add the connected user "${USER}" to the docker group.
    # Change the user name to match your preferred user.
    # You may have to logout and log back in again for
    # this to take effect.
    $ sudo gpasswd -a ${USER} docker

    # Restart the Docker daemon.
    $ sudo service docker restart


## What next?

Continue with the [User Guide](/userguide/).
