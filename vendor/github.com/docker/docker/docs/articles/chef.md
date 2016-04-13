<!--[metadata]>
+++
title = "Using Chef"
description = "Installation and using Docker via Chef"
keywords = ["chef, installation, usage, docker,  documentation"]
[menu.main]
parent = "smn_third_party"
+++
<![end-metadata]-->

# Using Chef

> **Note**:
> Please note this is a community contributed installation path. The only
> `official` installation is using the
> [*Ubuntu*](/installation/ubuntulinux) installation
> path. This version may sometimes be out of date.

## Requirements

To use this guide you'll need a working installation of
[Chef](http://www.getchef.com/). This cookbook supports a variety of
operating systems.

## Installation

The cookbook is available on the [Chef Community
Site](http://community.opscode.com/cookbooks/docker) and can be
installed using your favorite cookbook dependency manager.

The source can be found on
[GitHub](https://github.com/bflad/chef-docker).

## Usage

The cookbook provides recipes for installing Docker, configuring init
for Docker, and resources for managing images and containers. It
supports almost all Docker functionality.

### Installation

    include_recipe 'docker'

### Images

The next step is to pull a Docker image. For this, we have a resource:

    docker_image 'samalba/docker-registry'

This is equivalent to running:

    $ docker pull samalba/docker-registry

There are attributes available to control how long the cookbook will
allow for downloading (5 minute default).

To remove images you no longer need:

    docker_image 'samalba/docker-registry' do
      action :remove
    end

### Containers

Now you have an image where you can run commands within a container
managed by Docker.

    docker_container 'samalba/docker-registry' do
      detach true
      port '5000:5000'
      env 'SETTINGS_FLAVOR=local'
      volume '/mnt/docker:/docker-storage'
    end

This is equivalent to running the following command, but under upstart:

    $ docker run --detach=true --publish='5000:5000' --env='SETTINGS_FLAVOR=local' --volume='/mnt/docker:/docker-storage' samalba/docker-registry

The resources will accept a single string or an array of values for any
Docker flags that allow multiple values.
