<!--[metadata]>
+++
title = "Mirroring Docker Hub"
description = "Setting-up a local mirror for Docker Hub images"
keywords = ["registry, on-prem, images, tags, repository, distribution, mirror, Hub, recipe, advanced"]
+++
<![end-metadata]-->

# Registry as a pull through cache

## Use-case

If you have multiple instances of Docker running in your environment (e.g., multiple physical or virtual machines, all running the Docker daemon), each time one of them requires an image that it doesn’t have it will go out to the internet and fetch it from the public Docker registry. By running a local registry mirror, you can keep most of the redundant image fetch traffic on your local network.

### Alternatives

Alternatively, if the set of images you are using is well delimited, you can simply pull them manually and push them to a simple, local, private registry.

Furthermore, if your images are all built in-house, not using the Hub at all and relying entirely on your local registry is the simplest scenario.

### Gotcha

It's currently not possible to mirror another private registry. Only the central Hub can be mirrored.

### Solution

The Registry can be configured as a pull through cache.  In this mode a Registry responds to all normal docker pull requests but stores all content locally.

## How does it work?

The first time you request an image from your local registry mirror, it pulls the image from the public Docker registry and stores it locally before handing it back to you. On subsequent requests, the local registry mirror is able to serve the image from its own storage.

### What if the content changes on the Hub?

When a pull is attempted with a tag, the Registry will check the remote to ensure if it has the latest version of the requested content.  If it doesn't it will fetch the latest content and cache it.

### What about my disk?

In environments with high churn rates, stale data can build up in the cache.  When running as a pull through cache the Registry will periodically remove old content to save disk space. Subsequent requests for removed content will cause a remote fetch and local re-caching.

To ensure best performance and guarantee correctness the Registry cache should be configured to use the `filesystem` driver for storage.

## Running a Registry as a pull through cache

The easiest way to run a registry as a pull through cache is to run the official Registry image.

Multiple registry caches can be deployed over the same back-end.  A single registry cache will ensure that concurrent requests do not pull duplicate data, but this property will not hold true for a registry cache cluster.

### Configuring the cache

To configure a Registry to run as a pull through cache, the addition of a `proxy` section is required to the config file.

In order to access private images on the Docker Hub, a username and password can be supplied.

    proxy:
      remoteurl: https://registry-1.docker.io
      username: [username]
      password: [password]

> :warn: if you specify a username and password, it's very important to understand that private resources that this user has access to on the Hub will be made available on your mirror. It's thus paramount that you secure your mirror by implementing authentication if you expect these resources to stay private!

### Configuring the Docker daemon

You will need to pass the `--registry-mirror` option to your Docker daemon on startup:

    docker --registry-mirror=https://<my-docker-mirror-host> daemon

For example, if your mirror is serving on http://10.0.0.2:5000, you would run:

    docker --registry-mirror=https://10.0.0.2:5000 daemon

NOTE: Depending on your local host setup, you may be able to add the `--registry-mirror` option to the `DOCKER_OPTS` variable in `/etc/default/docker`.
