<!--[metadata]>
+++
title = "Installation on FrugalWare"
description = "Installation instructions for Docker on FrugalWare."
keywords = ["frugalware linux, virtualization, docker, documentation,  installation"]
[menu.main]
parent = "smn_linux"
+++
<![end-metadata]-->

# FrugalWare

Installing on FrugalWare is handled via the official packages:

 - [lxc-docker i686](http://www.frugalware.org/packages/200141)
 - [lxc-docker x86_64](http://www.frugalware.org/packages/200130)

The lxc-docker package will install the latest tagged version of Docker.

## Dependencies

Docker depends on several packages which are specified as dependencies
in the packages. The core dependencies are:

 - systemd
 - lvm2
 - sqlite3
 - libguestfs
 - lxc
 - iproute2
 - bridge-utils

## Installation

A simple

    $ sudo pacman -S lxc-docker

is all that is needed.

## Starting Docker

There is a systemd service unit created for Docker. To start Docker as
service:

    $ sudo systemctl start lxc-docker

To start on system boot:

    $ sudo systemctl enable lxc-docker

## Custom daemon options

If you need to add an HTTP Proxy, set a different directory or partition for the
Docker runtime files, or make other customizations, read our systemd article to
learn how to [customize your systemd Docker daemon options](/articles/systemd/).

## Uninstallation

To uninstall the Docker package:

    $ sudo pacman -R lxc-docker

To uninstall the Docker package and dependencies that are no longer needed:

    $ sudo pacman -Rns lxc-docker

The above commands will not remove images, containers, volumes, or user created
configuration files on your host. If you wish to delete all images, containers,
and volumes run the following command:

    $ rm -rf /var/lib/docker

You must delete the user created configuration files manually.
