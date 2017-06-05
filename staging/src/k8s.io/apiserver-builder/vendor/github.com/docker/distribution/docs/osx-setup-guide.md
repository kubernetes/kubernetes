<!--[metadata]>
+++
title = "Running on OS X"
description = "Explains how to run a registry on OS X"
keywords = ["registry, on-prem, images, tags, repository, distribution, OS X, recipe, advanced"]
+++
<![end-metadata]-->

# OS X Setup Guide

## Use-case

This is useful if you intend to run a registry server natively on OS X.

### Alternatives

You can start a VM on OS X, and deploy your registry normally as a container using Docker inside that VM.

The simplest road to get there is traditionally to use the [docker Toolbox](https://www.docker.com/toolbox), or [docker-machine](https://docs.docker.com/machine/), which usually relies on the [boot2docker](http://boot2docker.io/) iso inside a VirtualBox VM.

### Solution

Using the method described here, you install and compile your own from the git repository and run it as an OS X agent.

### Gotchas

Production services operation on OS X is out of scope of this document. Be sure you understand well these aspects before considering going to production with this.

## Setup golang on your machine

If you know, safely skip to the next section.

If you don't, the TLDR is:

    bash < <(curl -s -S -L https://raw.githubusercontent.com/moovweb/gvm/master/binscripts/gvm-installer)
    source ~/.gvm/scripts/gvm
    gvm install go1.4.2
    gvm use go1.4.2

If you want to understand, you should read [How to Write Go Code](https://golang.org/doc/code.html).

## Checkout the Docker Distribution source tree

    mkdir -p $GOPATH/src/github.com/docker
    git clone https://github.com/docker/distribution.git $GOPATH/src/github.com/docker/distribution
    cd $GOPATH/src/github.com/docker/distribution

## Build the binary

    GOPATH=$(PWD)/Godeps/_workspace:$GOPATH make binaries
    sudo cp bin/registry /usr/local/libexec/registry

## Setup

Copy the registry configuration file in place:

    mkdir /Users/Shared/Registry
    cp docs/osx/config.yml /Users/Shared/Registry/config.yml

## Running the Docker Registry under launchd

Copy the Docker registry plist into place:

    plutil -lint docs/osx/com.docker.registry.plist
    cp docs/osx/com.docker.registry.plist ~/Library/LaunchAgents/
    chmod 644 ~/Library/LaunchAgents/com.docker.registry.plist

Start the Docker registry:

    launchctl load ~/Library/LaunchAgents/com.docker.registry.plist

### Restarting the docker registry service

    launchctl stop com.docker.registry
    launchctl start com.docker.registry

### Unloading the docker registry service

    launchctl unload ~/Library/LaunchAgents/com.docker.registry.plist
