<!--[metadata]>
+++
title = "Run a local registry mirror"
description = "How to set up and run a local registry mirror"
keywords = ["docker, registry, mirror,  examples"]
[menu.main]
parent = "mn_docker_hub"
weight = 8
+++
<![end-metadata]-->

# Run a local registry mirror

## Why?

If you have multiple instances of Docker running in your environment
(e.g., multiple physical or virtual machines, all running the Docker
daemon), each time one of them requires an image that it doesn't have
it will go out to the internet and fetch it from the public Docker
registry. By running a local registry mirror, you can keep most of the
image fetch traffic on your local network.

## How does it work?

The first time you request an image from your local registry mirror,
it pulls the image from the public Docker registry and stores it locally
before handing it back to you. On subsequent requests, the local registry
mirror is able to serve the image from its own storage.

## How do I set up a local registry mirror?

There are two steps to set up and use a local registry mirror.

### Step 1: Configure your Docker daemons to use the local registry mirror

You will need to pass the `--registry-mirror` option to your Docker daemon on
startup:

    docker --registry-mirror=http://<my-docker-mirror-host> -d

For example, if your mirror is serving on `http://10.0.0.2:5000`, you would run:

    docker --registry-mirror=http://10.0.0.2:5000 -d

**NOTE:**
Depending on your local host setup, you may be able to add the
`--registry-mirror` options to the `DOCKER_OPTS` variable in
`/etc/default/docker`.

### Step 2: Run the local registry mirror

You will need to start a local registry mirror service. The
[`registry` image](https://registry.hub.docker.com/_/registry/) provides this
functionality. For example, to run a local registry mirror that serves on
port `5000` and mirrors the content at `registry-1.docker.io`:

    docker run -p 5000:5000 \
        -e STANDALONE=false \
        -e MIRROR_SOURCE=https://registry-1.docker.io \
        -e MIRROR_SOURCE_INDEX=https://index.docker.io \
        registry

## Test it out

With your mirror running, pull an image that you haven't pulled before (using
`time` to time it):

    $ time docker pull node:latest
    Pulling repository node
    [...]
    
    real   1m14.078s
    user   0m0.176s
    sys    0m0.120s

Now, remove the image from your local machine:

    $ docker rmi node:latest

Finally, re-pull the image:

    $ time docker pull node:latest
    Pulling repository node
    [...]
    
    real   0m51.376s
    user   0m0.120s
    sys    0m0.116s

The second time around, the local registry mirror served the image from storage,
avoiding a trip out to the internet to refetch it.
