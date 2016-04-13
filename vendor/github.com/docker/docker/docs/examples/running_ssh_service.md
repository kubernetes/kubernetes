<!--[metadata]>
+++
title = "Dockerizing an SSH service"
description = "Installing and running an SSHd service on Docker"
keywords = ["docker, example, package installation,  networking"]
[menu.main]
parent = "smn_apps_servs"
+++
<![end-metadata]-->

# Dockerizing an SSH daemon service

## Build an `eg_sshd` image

The following `Dockerfile` sets up an SSHd service in a container that you
can use to connect to and inspect other container's volumes, or to get
quick access to a test container.

    # sshd
    #
    # VERSION               0.0.2

    FROM ubuntu:14.04
    MAINTAINER Sven Dowideit <SvenDowideit@docker.com>

    RUN apt-get update && apt-get install -y openssh-server
    RUN mkdir /var/run/sshd
    RUN echo 'root:screencast' | chpasswd
    RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config

    # SSH login fix. Otherwise user is kicked off after login
    RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

    ENV NOTVISIBLE "in users profile"
    RUN echo "export VISIBLE=now" >> /etc/profile

    EXPOSE 22
    CMD ["/usr/sbin/sshd", "-D"]

Build the image using:

    $ docker build -t eg_sshd .

## Run a `test_sshd` container

Then run it. You can then use `docker port` to find out what host port
the container's port 22 is mapped to:

    $ docker run -d -P --name test_sshd eg_sshd
    $ docker port test_sshd 22
    0.0.0.0:49154

And now you can ssh as `root` on the container's IP address (you can find it
with `docker inspect`) or on port `49154` of the Docker daemon's host IP address
(`ip address` or `ifconfig` can tell you that) or `localhost` if on the
Docker daemon host:

    $ ssh root@192.168.1.2 -p 49154
    # The password is ``screencast``.
    $$

## Environment variables

Using the `sshd` daemon to spawn shells makes it complicated to pass environment
variables to the user's shell via the normal Docker mechanisms, as `sshd` scrubs
the environment before it starts the shell.

If you're setting values in the `Dockerfile` using `ENV`, you'll need to push them
to a shell initialization file like the `/etc/profile` example in the `Dockerfile`
above.

If you need to pass`docker run -e ENV=value` values, you will need to write a
short script to do the same before you start `sshd -D` and then replace the
`CMD` with that script.

## Clean up

Finally, clean up after your test by stopping and removing the
container, and then removing the image.

    $ docker stop test_sshd
    $ docker rm test_sshd
    $ docker rmi eg_sshd

