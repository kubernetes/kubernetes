<!--[metadata]>
+++
title = "Control and configure Docker with systemd"
description = "Controlling and configuring Docker using systemd"
keywords = ["docker, daemon, systemd,  configuration"]
[menu.main]
parent = "smn_administrate"
weight = 7
+++
<![end-metadata]-->

# Control and configure Docker with systemd

Many Linux distributions use systemd to start the Docker daemon. This document
shows a few examples of how to customise Docker's settings.

## Starting the Docker daemon

Once Docker is installed, you will need to start the Docker daemon.

    $ sudo systemctl start docker
    # or on older distributions, you may need to use
    $ sudo service docker start

If you want Docker to start at boot, you should also:

    $ sudo systemctl enable docker
    # or on older distributions, you may need to use
    $ sudo chkconfig docker on

## Custom Docker daemon options

There are a number of ways to configure the daemon flags and environment variables
for your Docker daemon.

If the `docker.service` file is set to use an `EnvironmentFile`
(often pointing to `/etc/sysconfig/docker`) then you can modify the
referenced file.

Check if the `docker.service` uses an `EnvironmentFile`:

    $ sudo systemctl show docker | grep EnvironmentFile
    EnvironmentFile=-/etc/sysconfig/docker (ignore_errors=yes)

Alternatively, find out where the service file is located, and look for the
property:

    $ sudo systemctl status docker | grep Loaded
       Loaded: loaded (/usr/lib/systemd/system/docker.service; enabled)
    $ sudo grep EnvironmentFile /usr/lib/systemd/system/docker.service
    EnvironmentFile=-/etc/sysconfig/docker

You can customize the Docker daemon options using override files as explained in the
[HTTP Proxy example](#http-proxy) below. The files located in `/usr/lib/systemd/system` 
or `/lib/systemd/system` contain the default options and should not be edited.

### Runtime directory and storage driver

You may want to control the disk space used for Docker images, containers
and volumes by moving it to a separate partition.

In this example, we'll assume that your `docker.service` file looks something like:

    [Unit]
    Description=Docker Application Container Engine
    Documentation=https://docs.docker.com
    After=network.target docker.socket
    Requires=docker.socket

    [Service]
    Type=notify
    EnvironmentFile=-/etc/sysconfig/docker
    ExecStart=/usr/bin/docker -d -H fd:// $OPTIONS
    LimitNOFILE=1048576
    LimitNPROC=1048576

    [Install]
    Also=docker.socket

This will allow us to add extra flags to the `/etc/sysconfig/docker` file by
setting `OPTIONS`:

    OPTIONS="--graph /mnt/docker-data --storage-driver btrfs"

You can also set other environment variables in this file, for example, the
`HTTP_PROXY` environment variables described below.

### HTTP proxy

This example overrides the default `docker.service` file.

If you are behind a HTTP proxy server, for example in corporate settings,
you will need to add this configuration in the Docker systemd service file.

First, create a systemd drop-in directory for the docker service:

    mkdir /etc/systemd/system/docker.service.d

Now create a file called `/etc/systemd/system/docker.service.d/http-proxy.conf`
that adds the `HTTP_PROXY` environment variable:

    [Service]
    Environment="HTTP_PROXY=http://proxy.example.com:80/"

If you have internal Docker registries that you need to contact without
proxying you can specify them via the `NO_PROXY` environment variable:

    Environment="HTTP_PROXY=http://proxy.example.com:80/" "NO_PROXY=localhost,127.0.0.0/8,docker-registry.somecorporation.com"

Flush changes:

    $ sudo systemctl daemon-reload

Verify that the configuration has been loaded:

    $ sudo systemctl show docker --property Environment
    Environment=HTTP_PROXY=http://proxy.example.com:80/

Restart Docker:

    $ sudo systemctl restart docker

## Manually creating the systemd unit files

When installing the binary without a package, you may want
to integrate Docker with systemd. For this, simply install the two unit files
(service and socket) from [the github
repository](https://github.com/docker/docker/tree/master/contrib/init/systemd)
to `/etc/systemd/system`.


