<!--[metadata]>
+++
title = "Getting started with Docker Hub"
description = "Introductory guide to getting an account on Docker Hub"
keywords = ["documentation, docs, the docker guide, docker guide, docker, docker platform, virtualization framework, docker.io, central service, services, how to, container, containers, automation, collaboration, collaborators, registry, repo, repository, technology, github webhooks,  trusted builds"]
[menu.main]
parent = "smn_pubhub"
weight = 1
+++
<![end-metadata]-->

# Getting started with Docker Hub


This section provides a quick introduction to the [Docker Hub](https://hub.docker.com),
including how to create an account.

The [Docker Hub](https://hub.docker.com) is a centralized resource for working with
Docker and its components. Docker Hub helps you collaborate with colleagues and get the
most out of Docker. To do this, it provides services such as:

* Docker image hosting.
* User authentication.
* Automated image builds and work-flow tools such as build triggers and web
  hooks.
* Integration with GitHub and Bitbucket.

In order to use Docker Hub, you will first need to register and create an account. Don't
worry, creating an account is simple and free.

## Creating a Docker Hub account

There are two ways for you to register and create an account:

1. Via the web, or
2. Via the command line.

### Register via the web

Fill in the [sign-up form](https://hub.docker.com/account/signup/) by
choosing your user name and password and entering a valid email address. You can also
sign up for the Docker Weekly mailing list, which has lots of information about what's
going on in the world of Docker.

![Register using the sign-up page](/userguide/register-web.png)

### Register via the command line

You can also create a Docker Hub account via the command line with the
`docker login` command.

    $ docker login

### Confirm your email

Once you've filled in the form, check your email for a welcome message asking for
confirmation so we can activate your account.


### Login

After you complete the confirmation process, you can login using the web console:

![Login using the web console](/userguide/login-web.png)

Or via the command line with the `docker login` command:

    $ docker login

Your Docker Hub account is now active and ready to use.

##  Next steps

Next, let's start learning how to Dockerize applications with our "Hello world"
exercise.

Go to [Dockerizing Applications](/userguide/dockerizing).

