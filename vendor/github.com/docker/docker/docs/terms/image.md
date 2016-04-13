<!--[metadata]>
+++
draft = true 
title = "Image"
description = "Definition of an image"
keywords = ["containers, lxc, concepts, explanation, image,  container"]
[menu.main]
parent = "mn_reference"
+++
<![end-metadata]-->

# Image

## Introduction

![](/terms/images/docker-filesystems-debian.png)

In Docker terminology, a read-only [*Layer*](/terms/layer/#layer) is
called an **image**. An image never changes.

Since Docker uses a [*Union File System*](/terms/layer/#union-file-system), the
processes think the whole file system is mounted read-write. But all the
changes go to the top-most writeable layer, and underneath, the original
file in the read-only image is unchanged. Since images don't change,
images do not have state.

![](/terms/images/docker-filesystems-debianrw.png)

## Parent image

![](/terms/images/docker-filesystems-multilayer.png)

Each image may depend on one more image which forms the layer beneath
it. We sometimes say that the lower image is the **parent** of the upper
image.

## Base image

An image that has no parent is a **base image**.

## Image IDs

All images are identified by a 64 hexadecimal digit string (internally a
256bit value). To simplify their use, a short ID of the first 12
characters can be used on the command line. There is a small possibility
of short id collisions, so the docker server will always return the long
ID.
