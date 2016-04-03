<!--[metadata]>
+++
title = "save"
description = "The save command description and usage"
keywords = ["tarred, repository, backup"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# save

    Usage: docker save [OPTIONS] IMAGE [IMAGE...]

    Save an image(s) to a tar archive (streamed to STDOUT by default)

      -o, --output=""    Write to a file, instead of STDOUT

Produces a tarred repository to the standard output stream.
Contains all parent layers, and all tags + versions, or specified `repo:tag`, for
each argument provided.

It is used to create a backup that can then be used with `docker load`

    $ docker save busybox > busybox.tar
    $ ls -sh busybox.tar
    2.7M busybox.tar
    $ docker save --output busybox.tar busybox
    $ ls -sh busybox.tar
    2.7M busybox.tar
    $ docker save -o fedora-all.tar fedora
    $ docker save -o fedora-latest.tar fedora:latest

It is even useful to cherry-pick particular tags of an image repository

    $ docker save -o ubuntu.tar ubuntu:lucid ubuntu:saucy
