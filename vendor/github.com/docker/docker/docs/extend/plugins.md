<!--[metadata]>
+++
title = "Extending Docker with plugins"
description = "How to add additional functionality to Docker with plugins extensions"
keywords = ["Examples, Usage, plugins, docker, documentation, user guide"]
[menu.main]
parent = "mn_extend"
weight=-1
+++
<![end-metadata]-->

# Understand Docker plugins

You can extend the capabilities of the Docker Engine by loading third-party
plugins.

## Types of plugins

Plugins extend Docker's functionality.  They come in specific types.  For
example, a [volume plugin](plugins_volume.md) might enable Docker
volumes to persist across multiple Docker hosts.

Currently Docker supports volume and network driver plugins. In the future it
will support additional plugin types.

## Installing a plugin

Follow the instructions in the plugin's documentation.

## Finding a plugin

The following plugins exist:

* The [Flocker plugin](https://clusterhq.com/docker-plugin/) is a volume plugin
  which provides multi-host portable volumes for Docker, enabling you to run
  databases and other stateful containers and move them around across a cluster
  of machines.

* The [GlusterFS plugin](https://github.com/calavera/docker-volume-glusterfs) is
  another volume plugin that provides multi-host volumes management for Docker
  using GlusterFS.

* The [Keywhiz plugin](https://github.com/calavera/docker-volume-keywhiz) is
  a plugin that provides credentials and secret management using Keywhiz as
  a central repository.

## Troubleshooting a plugin

If you are having problems with Docker after loading a plugin, ask the authors
of the plugin for help. The Docker team may not be able to assist you.

## Writing a plugin

If you are interested in writing a plugin for Docker, or seeing how they work
under the hood, see the [docker plugins reference](plugin_api.md).
