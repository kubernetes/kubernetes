<!--[metadata]>
+++
title = "Installation on CRUX Linux"
description = "Docker installation on CRUX Linux."
keywords = ["crux linux, virtualization, Docker, documentation,  installation"]
[menu.main]
parent = "smn_linux"
+++
<![end-metadata]-->

# CRUX Linux

Installing on CRUX Linux can be handled via the contrib ports from
[James Mills](http://prologic.shortcircuit.net.au/) and are included in the
official [contrib](http://crux.nu/portdb/?a=repo&q=contrib) ports:

- docker

The `docker` port will build and install the latest tagged version of Docker.


## Installation

Assuming you have contrib enabled, update your ports tree and install docker:

    $ sudo prt-get depinst docker


## Kernel requirements

To have a working **CRUX+Docker** Host you must ensure your Kernel has
the necessary modules enabled for the Docker Daemon to function correctly.

Please read the `README`:

    $ sudo prt-get readme docker

The `docker` port installs the `contrib/check-config.sh` script
provided by the Docker contributors for checking your kernel
configuration as a suitable Docker host.

To check your Kernel configuration run:

    $ /usr/share/docker/check-config.sh

## Starting Docker

There is a rc script created for Docker. To start the Docker service:

    $ sudo /etc/rc.d/docker start

To start on system boot:

 - Edit `/etc/rc.conf`
 - Put `docker` into the `SERVICES=(...)` array after `net`.

## Images

There is a CRUX image maintained by [James Mills](http://prologic.shortcircuit.net.au/)
as part of the Docker "Official Library" of images. To use this image simply pull it
or use it as part of your `FROM` line in your `Dockerfile(s)`.

    $ docker pull crux
    $ docker run -i -t crux

There are also user contributed [CRUX based image(s)](https://registry.hub.docker.com/repos/crux/) on the Docker Hub.


## Uninstallation

To uninstall the Docker package:

    $ sudo prt-get remove docker

The above command will not remove images, containers, volumes, or user created
configuration files on your host. If you wish to delete all images, containers,
and volumes run the following command:

    $ rm -rf /var/lib/docker

You must delete the user created configuration files manually.

## Issues

If you have any issues please file a bug with the
[CRUX Bug Tracker](http://crux.nu/bugs/).

## Support

For support contact the [CRUX Mailing List](http://crux.nu/Main/MailingLists)
or join CRUX's [IRC Channels](http://crux.nu/Main/IrcChannels). on the
[FreeNode](http://freenode.net/) IRC Network.
