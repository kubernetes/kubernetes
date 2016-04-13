<!--[metadata]>
+++
title = "Get the required software"
description = "Describes the software required to contribute to Docker"
keywords = ["GitHub account, repository, Docker, Git, Go, make,  "]
[menu.main]
parent = "smn_develop"
weight=2
+++
<![end-metadata]-->

# Get the required software for Linux or OS X

This page explains how to get the software you need to use a Linux or OS X
machine for Docker development. Before you begin contributing you must have:

*  a GitHub account
* `git`
* `make` 
* `docker`

You'll notice that `go`, the language that Docker is written in, is not listed.
That's because you don't need it installed; Docker's development environment
provides it for you. You'll learn more about the development environment later.

### Get a GitHub account

To contribute to the Docker project, you will need a <a
href="https://github.com" target="_blank">GitHub account</a>. A free account is
fine. All the Docker project repositories are public and visible to everyone.

You should also have some experience using both the GitHub application and `git`
on the command line. 

### Install git

Install `git` on your local system. You can check if `git` is on already on your
system and properly installed with the following command:

    $ git --version 


This documentation is written using `git` version 2.2.2. Your version may be
different depending on your OS.

### Install make

Install `make`. You can check if `make` is on your system with the following
command:

    $ make -v 

This documentation is written using GNU Make 3.81. Your version may be different
depending on your OS.

### Install or upgrade Docker 

If you haven't already, install the Docker software using the 
<a href="/installation" target="_blank">instructions for your operating system</a>.
If you have an existing installation, check your version and make sure you have
the latest Docker. 

To check if `docker` is already installed on Linux:

    $ docker --version
    Docker version 1.5.0, build a8a31ef

On Mac OS X or Windows, you should have installed Boot2Docker which includes
Docker. You'll need to verify both Boot2Docker and then Docker. This
documentation was written on OS X using the following versions.

    $ boot2docker version
    Boot2Docker-cli version: v1.5.0
    Git commit: ccd9032

    $ docker --version
    Docker version 1.5.0, build a8a31ef

## Linux users and sudo

This guide assumes you have added your user to the `docker` group on your system.
To check, list the group's contents:

    $ getent group docker
    docker:x:999:ubuntu

If the command returns no matches, you have two choices. You can preface this
guide's `docker` commands with `sudo` as you work. Alternatively, you can add
your user to the `docker` group as follows:

    $ sudo usermod -aG docker ubuntu

You must log out and log back in for this modification to take effect.


## Where to go next

In the next section, you'll [learn how to set up and configure Git for
contributing to Docker](/project/set-up-git/).
