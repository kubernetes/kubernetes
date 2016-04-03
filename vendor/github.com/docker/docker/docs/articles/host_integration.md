<!--[metadata]>
+++
title = "Automatically start containers"
description = "How to generate scripts for upstart, systemd, etc."
keywords = ["systemd, upstart, supervisor, docker, documentation,  host integration"]
[menu.main]
parent = "smn_containers"
weight = 99
+++
<![end-metadata]-->

# Automatically start containers

As of Docker 1.2,
[restart policies](/reference/commandline/cli/#restart-policies) are the
built-in Docker mechanism for restarting containers when they exit. If set,
restart policies will be used when the Docker daemon starts up, as typically
happens after a system boot. Restart policies will ensure that linked containers
are started in the correct order.

If restart policies don't suit your needs (i.e., you have non-Docker processes
that depend on Docker containers), you can use a process manager like
[upstart](http://upstart.ubuntu.com/),
[systemd](http://freedesktop.org/wiki/Software/systemd/) or
[supervisor](http://supervisord.org/) instead.


## Using a process manager

Docker does not set any restart policies by default, but be aware that they will
conflict with most process managers. So don't set restart policies if you are
using a process manager.

*Note:* Prior to Docker 1.2, restarting of Docker containers had to be
explicitly disabled. Refer to the
[previous version](/v1.1/articles/host_integration/) of this article for the
details on how to do that.

When you have finished setting up your image and are happy with your
running container, you can then attach a process manager to manage it.
When you run `docker start -a`, Docker will automatically attach to the
running container, or start it if needed and forward all signals so that
the process manager can detect when a container stops and correctly
restart it.

Here are a few sample scripts for systemd and upstart to integrate with
Docker.


## Examples

The examples below show configuration files for two popular process managers,
upstart and systemd. In these examples, we'll assume that we have already
created a container to run Redis with `--name=redis_server`. These files define
a new service that will be started after the docker daemon service has started.


### upstart

    description "Redis container"
    author "Me"
    start on filesystem and started docker
    stop on runlevel [!2345]
    respawn
    script
      /usr/bin/docker start -a redis_server
    end script

### systemd

    [Unit]
    Description=Redis container
    Requires=docker.service
    After=docker.service

    [Service]
    Restart=always
    ExecStart=/usr/bin/docker start -a redis_server
    ExecStop=/usr/bin/docker stop -t 2 redis_server

    [Install]
    WantedBy=local.target

If you need to pass options to the redis container (such as `--env`),
then you'll need to use `docker run` rather than `docker start`. This will
create a new container every time the service is started, which will be stopped
and removed when the service is stopped.

    [Service]
    ...
    ExecStart=/usr/bin/docker run --env foo=bar --name redis_server redis
    ExecStop=/usr/bin/docker stop -t 2 redis_server ; /usr/bin/docker rm -f redis_server
    ...
