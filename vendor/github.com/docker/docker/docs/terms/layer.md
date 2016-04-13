<!--[metadata]>
+++
draft = true
title = "Layers"
description = "Organizing the Docker Root File System"
keywords = ["containers, lxc, concepts, explanation, image,  container"]
[menu.main]
parent = "mn_use_docker"
+++
<![end-metadata]-->

# Layers

## Introduction

In a traditional Linux boot, the kernel first mounts the root [*File
System*](/terms/filesystem) as read-only, checks its
integrity, and then switches the whole rootfs volume to read-write mode.

## Layer

When Docker mounts the rootfs, it starts read-only, as in a traditional
Linux boot, but then, instead of changing the file system to read-write
mode, it takes advantage of a [union
mount](http://en.wikipedia.org/wiki/Union_mount) to add a read-write
file system *over* the read-only file system. In fact there may be
multiple read-only file systems stacked on top of each other. We think
of each one of these file systems as a **layer**.

![](/terms/images/docker-filesystems-multilayer.png)

At first, the top read-write layer has nothing in it, but any time a
process creates a file, this happens in the top layer. And if something
needs to update an existing file in a lower layer, then the file gets
copied to the upper layer and changes go into the copy. The version of
the file on the lower layer cannot be seen by the applications anymore,
but it is there, unchanged.

## Union File System

We call the union of the read-write layer and all the read-only layers a
**union file system**.
