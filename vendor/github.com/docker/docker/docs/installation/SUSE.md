<!--[metadata]>
+++
title = "Installation on openSUSE and SUSE Linux Enterprise"
description = "Installation instructions for Docker on openSUSE and on SUSE Linux Enterprise."
keywords = ["openSUSE, SUSE Linux Enterprise, SUSE, SLE, docker, documentation,  installation"]
[menu.main]
parent = "smn_linux"
+++
<![end-metadata]-->

# openSUSE

Docker is available in **openSUSE 12.3 and later**. Please note that due
to its current limitations Docker is able to run only **64 bit** architecture.

Docker is not part of the official repositories of openSUSE 12.3 and
openSUSE 13.1. Hence  it is necessary to add the [Virtualization
repository](https://build.opensuse.org/project/show/Virtualization) from
[OBS](https://build.opensuse.org/) to install the `docker` package.

Execute one of the following commands to add the Virtualization repository:

    # openSUSE 12.3
    $ sudo zypper ar -f http://download.opensuse.org/repositories/Virtualization/openSUSE_12.3/ Virtualization

    # openSUSE 13.1
    $ sudo zypper ar -f http://download.opensuse.org/repositories/Virtualization/openSUSE_13.1/ Virtualization

No extra repository is required for openSUSE 13.2 and later.

# SUSE Linux Enterprise

Docker is available in **SUSE Linux Enterprise 12 and later**. Please note that
due to its current limitations Docker is able to run only on **64 bit**
architecture.

## Installation

Install the Docker package.

    $ sudo zypper in docker

Now that it's installed, let's start the Docker daemon.

    $ sudo systemctl start docker

If we want Docker to start at boot, we should also:

    $ sudo systemctl enable docker

The docker package creates a new group named docker. Users, other than
root user, need to be part of this group in order to interact with the
Docker daemon. You can add users with:

    $ sudo /usr/sbin/usermod -a -G docker <username>

To verify that everything has worked as expected:

    $ sudo docker run --rm -i -t opensuse /bin/bash

This should download and import the `opensuse` image, and then start `bash` in
a container. To exit the container type `exit`.

If you want your containers to be able to access the external network you must
enable the `net.ipv4.ip_forward` rule.
This can be done using YaST by browsing to the
`Network Devices -> Network Settings -> Routing` menu and ensuring that the
`Enable IPv4 Forwarding` box is checked.

This option cannot be changed when networking is handled by the Network Manager.
In such cases the `/etc/sysconfig/SuSEfirewall2` file needs to be edited by
hand to ensure the `FW_ROUTE` flag is set to `yes` like so:

    FW_ROUTE="yes"


**Done!**

## Custom daemon options

If you need to add an HTTP Proxy, set a different directory or partition for the
Docker runtime files, or make other customizations, read our systemd article to
learn how to [customize your systemd Docker daemon options](/articles/systemd/).

## Uninstallation

To uninstall the Docker package:

    $ sudo zypper rm docker

The above command will not remove images, containers, volumes, or user created
configuration files on your host. If you wish to delete all images, containers,
and volumes run the following command:

    $ rm -rf /var/lib/docker

You must delete the user created configuration files manually.

## What's next

Continue with the [User Guide](/userguide/).

