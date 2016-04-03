<!--[metadata]>
+++
title = "Dockerizing MongoDB"
description = "Creating a Docker image with MongoDB pre-installed using a Dockerfile and sharing the image on Docker Hub"
keywords = ["docker, dockerize, dockerizing, article, example, docker.io, platform, package, installation, networking, mongodb, containers, images, image, sharing, dockerfile, build, auto-building, virtualization,  framework"]
[menu.main]
parent = "smn_applied"
+++
<![end-metadata]-->

# Dockerizing MongoDB

## Introduction

In this example, we are going to learn how to build a Docker image with
MongoDB pre-installed.  We'll also see how to `push` that image to the
[Docker Hub registry](https://hub.docker.com) and share it with others!

> **Note:**
>
> This guide will show the mechanics of building a MongoDB container, but
> you will probably want to use the official image on [Docker Hub]( https://registry.hub.docker.com/_/mongo/)

Using Docker and containers for deploying [MongoDB](https://www.mongodb.org/)
instances will bring several benefits, such as:

 - Easy to maintain, highly configurable MongoDB instances;
 - Ready to run and start working within milliseconds;
 - Based on globally accessible and shareable images.

> **Note:**
> 
> If you do **_not_** like `sudo`, you might want to check out: 
> [*Giving non-root access*](/installation/binaries/#giving-non-root-access).

## Creating a Dockerfile for MongoDB

Let's create our `Dockerfile` and start building it:

    $ nano Dockerfile

Although optional, it is handy to have comments at the beginning of a
`Dockerfile` explaining its purpose:

    # Dockerizing MongoDB: Dockerfile for building MongoDB images
    # Based on ubuntu:latest, installs MongoDB following the instructions from:
    # http://docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/

> **Tip:** `Dockerfile`s are flexible. However, they need to follow a certain
> format. The first item to be defined is the name of an image, which becomes
> the *parent* of your *Dockerized MongoDB* image.

We will build our image using the latest version of Ubuntu from the
[Docker Hub Ubuntu](https://registry.hub.docker.com/_/ubuntu/) repository.

    # Format: FROM    repository[:version]
    FROM       ubuntu:latest

Continuing, we will declare the `MAINTAINER` of the `Dockerfile`:

    # Format: MAINTAINER Name <email@addr.ess>
    MAINTAINER M.Y. Name <myname@addr.ess>

> **Note:** Although Ubuntu systems have MongoDB packages, they are likely to
> be outdated. Therefore in this example, we will use the official MongoDB
> packages.

We will begin with importing the MongoDB public GPG key. We will also create
a MongoDB repository file for the package manager.

    # Installation:
    # Import MongoDB public GPG key AND create a MongoDB list file
    RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10
    RUN echo "deb http://repo.mongodb.org/apt/ubuntu "$(lsb_release -sc)"/mongodb-org/3.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-3.0.list

After this initial preparation we can update our packages and install MongoDB.

    # Update apt-get sources AND install MongoDB
    RUN apt-get update && apt-get install -y mongodb-org

> **Tip:** You can install a specific version of MongoDB by using a list
> of required packages with versions, e.g.:
> 
>     RUN apt-get update && apt-get install -y mongodb-org=3.0.1 mongodb-org-server=3.0.1 mongodb-org-shell=3.0.1 mongodb-org-mongos=3.0.1 mongodb-org-tools=3.0.1

MongoDB requires a data directory. Let's create it as the final step of our
installation instructions.

    # Create the MongoDB data directory
    RUN mkdir -p /data/db

Lastly we set the `ENTRYPOINT` which will tell Docker to run `mongod` inside
the containers launched from our MongoDB image. And for ports, we will use
the `EXPOSE` instruction.

    # Expose port 27017 from the container to the host
    EXPOSE 27017

    # Set usr/bin/mongod as the dockerized entry-point application
    ENTRYPOINT ["/usr/bin/mongod"]

Now save the file and let's build our image.

> **Note:**
> 
> The full version of this `Dockerfile` can be found [here](/examples/mongodb/Dockerfile).

## Building the MongoDB Docker image

With our `Dockerfile`, we can now build the MongoDB image using Docker. Unless
experimenting, it is always a good practice to tag Docker images by passing the
`--tag` option to `docker build` command.

    # Format: docker build --tag/-t <user-name>/<repository> .
    # Example:
    $ docker build --tag my/repo .

Once this command is issued, Docker will go through the `Dockerfile` and build
the image. The final image will be tagged `my/repo`.

## Pushing the MongoDB image to Docker Hub

All Docker image repositories can be hosted and shared on
[Docker Hub](https://hub.docker.com) with the `docker push` command. For this,
you need to be logged-in.

    # Log-in
    $ docker login
    Username:
    ..

    # Push the image
    # Format: docker push <user-name>/<repository>
    $ docker push my/repo
    The push refers to a repository [my/repo] (len: 1)
    Sending image list
    Pushing repository my/repo (1 tags)
    ..

## Using the MongoDB image

Using the MongoDB image we created, we can run one or more MongoDB instances
as daemon process(es).

    # Basic way
    # Usage: docker run --name <name for container> -d <user-name>/<repository>
    $ docker run -p 27017:27017 --name mongo_instance_001 -d my/repo

    # Dockerized MongoDB, lean and mean!
    # Usage: docker run --name <name for container> -d <user-name>/<repository> --noprealloc --smallfiles
    $ docker run -p 27017:27017 --name mongo_instance_001 -d my/repo --noprealloc --smallfiles

    # Checking out the logs of a MongoDB container
    # Usage: docker logs <name for container>
    $ docker logs mongo_instance_001

    # Playing with MongoDB
    # Usage: mongo --port <port you get from `docker ps`> 
    $ mongo --port 27017

    # If using boot2docker
    # Usage: mongo --port <port you get from `docker ps`>  --host <ip address from `boot2docker ip`>
    $ mongo --port 27017 --host 192.168.59.103

> **Tip:**
If you want to run two containers on the same engine, then you will need to map
the exposed port to two different ports on the host

    # Start two containers and map the ports
    $ docker run -p 28001:27017 --name mongo_instance_001 -d my/repo
    $ docker run -p 28002:27017 --name mongo_instance_002 -d my/repo

    # Now you can connect to each MongoDB instance on the two ports
    $ mongo --port 28001
    $ mongo --port 28002

 - [Linking containers](/userguide/dockerlinks)
 - [Cross-host linking containers](/articles/ambassador_pattern_linking/)
 - [Creating an Automated Build](/docker-io/builds/#automated-builds)
