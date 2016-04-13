<!--[metadata]>
+++
title = "Docker Hub user guide"
description = "Docker Hub user guide"
keywords = ["Docker, docker, registry, Docker Hub, docs,  documentation"]
[menu.main]
parent = "smn_pubhub"
+++
<![end-metadata]-->

# Using the Docker Hub

Docker Hub is used to find and pull Docker images to run or build upon, and to
distribute and build images for other users to use.

![your profile](/docker-hub/hub-images/dashboard.png)

## Finding repositories and images

There are two ways you can search for public repositories and images available
on the Docker Hub. You can use the "Search" tool on the Docker Hub website, or
you can `search` for all the repositories and images using the Docker commandline
tool:

    $ docker search ubuntu

Both will show you a list of the currently available public repositories on the
Docker Hub which match the provided keyword.

If a repository is private or marked as unlisted, it won't be in the repository
search results. To see all the repositories you have access to and their statuses,
you can look at your profile page on [Docker Hub](https://hub.docker.com).

## Pulling, running and building images

You can find more information on [working with Docker images](../userguide/dockerimages.md).

## Official Repositories

The Docker Hub contains a number of [Official
Repositories](http://registry.hub.docker.com/official). These are
certified repositories from vendors and contributors to Docker. They
contain Docker images from vendors like Canonical, Oracle, and Red Hat
that you can use to build applications and services.

If you use Official Repositories you know you're using an optimized and
up-to-date image to power your applications.

> **Note:**
> If you would like to contribute an Official Repository for your
> organization, see [Official Repositories on Docker
> Hub](/docker-hub/official_repos) for more information.

## Building and shipping your own repositories and images

The Docker Hub provides you and your team with a place to build and ship Docker images.

Collections of Docker images are managed using repositories - 

You can configure two types of repositories to manage on the Docker Hub:
[Repositories](./repos.md), which allow you to push images to the Hub from your local Docker daemon,
and [Automated Builds](./builds.md), which allow you to configure GitHub or Bitbucket to
trigger the Hub to rebuild repositories when changes are made to the repository.
