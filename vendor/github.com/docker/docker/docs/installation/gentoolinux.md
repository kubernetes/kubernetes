<!--[metadata]>
+++
title = "Installation on Gentoo"
description = "Installation instructions for Docker on Gentoo."
keywords = ["gentoo linux, virtualization, docker, documentation,  installation"]
[menu.main]
parent = "smn_linux"
+++
<![end-metadata]-->

# Gentoo

Installing Docker on Gentoo Linux can be accomplished using one of two ways: the **official** way and the `docker-overlay` way.

Official project page of [Gentoo Docker](https://wiki.gentoo.org/wiki/Project:Docker) team.

## Official way
The first and recommended way if you are looking for a stable  
experience is to use the official `app-emulation/docker` package directly  
from the tree.

If any issues arise from this ebuild including, missing kernel 
configuration flags or dependencies, open a bug 
on the Gentoo [Bugzilla](https://bugs.gentoo.org) assigned to `docker AT gentoo DOT org` 
or join and ask in the official
[IRC](http://webchat.freenode.net?channels=%23gentoo-containers&uio=d4) channel on the Freenode network.

## docker-overlay way

If you're looking for a `-bin` ebuild, a live ebuild, or a bleeding edge
ebuild, use the provided overlay, [docker-overlay](https://github.com/tianon/docker-overlay)
which can be added using `app-portage/layman`. The most accurate and
up-to-date documentation for properly installing and using the overlay
can be found in the [overlay](https://github.com/tianon/docker-overlay/blob/master/README.md#using-this-overlay).

If any issues arise from this ebuild or the resulting binary, including
and especially missing kernel configuration flags or dependencies, 
open an [issue](https://github.com/tianon/docker-overlay/issues) on 
the `docker-overlay` repository or ping `tianon` directly in the `#docker` 
IRC channel on the Freenode network.

## Installation

### Available USE flags

| USE Flag      | Default | Description |
| ------------- |:-------:|:------------|
| aufs          |         |Enables dependencies for the "aufs" graph driver, including necessary kernel flags.|
| btrfs         |         |Enables dependencies for the "btrfs" graph driver, including necessary kernel flags.|
| contrib       |  Yes    |Install additional contributed scripts and components.|
| device-mapper |  Yes    |Enables dependencies for the "devicemapper" graph driver, including necessary kernel flags.|
| doc           |         |Add extra documentation (API, Javadoc, etc). It is recommended to enable per package instead of globally.|
| lxc           |         |Enables dependencies for the "lxc" execution driver.|
| vim-syntax    |         |Pulls in related vim syntax scripts.|
| zsh-completion|         |Enable zsh completion support.|

USE flags are described in detail on [tianon's
blog](https://tianon.github.io/post/2014/05/17/docker-on-gentoo.html).

The package should properly pull in all the necessary dependencies and
prompt for all necessary kernel options.

    $ sudo emerge -av app-emulation/docker

>Note: Sometimes there is a disparity between the latest versions 
>in the official **Gentoo tree** and the **docker-overlay**.  
>Please be patient, and the latest version should propagate shortly.

## Starting Docker

Ensure that you are running a kernel that includes all the necessary
modules and configuration (and optionally for device-mapper
and AUFS or Btrfs, depending on the storage driver you've decided to use).

To use Docker, the `docker` daemon must be running as **root**.  
To use Docker as a **non-root** user, add yourself to the **docker** 
group by running the following command:

    $ sudo usermod -a -G docker user
 
### OpenRC

To start the `docker` daemon:

    $ sudo /etc/init.d/docker start

To start on system boot:

    $ sudo rc-update add docker default

### systemd

To start the `docker` daemon:

    $ sudo systemctl start docker

To start on system boot:

    $ sudo systemctl enable docker
   
If you need to add an HTTP Proxy, set a different directory or partition for the
Docker runtime files, or make other customizations, read our systemd article to
learn how to [customize your systemd Docker daemon options](/articles/systemd/).

## Uninstallation

To uninstall the Docker package:

    $ sudo emerge -cav app-emulation/docker

To uninstall the Docker package and dependencies that are no longer needed:

    $ sudo emerge -C app-emulation/docker

The above commands will not remove images, containers, volumes, or user created
configuration files on your host. If you wish to delete all images, containers,
and volumes run the following command:

    $ rm -rf /var/lib/docker

You must delete the user created configuration files manually.
