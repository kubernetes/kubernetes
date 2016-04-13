<!--[metadata]>
+++
title = "Process management with CFEngine"
description = "Managing containerized processes with CFEngine"
keywords = ["cfengine, process, management, usage, docker,  documentation"]
[menu.main]
parent = "smn_third_party"
+++
<![end-metadata]-->

# Process management with CFEngine

Create Docker containers with managed processes.

Docker monitors one process in each running container and the container
lives or dies with that process. By introducing CFEngine inside Docker
containers, we can alleviate a few of the issues that may arise:

 - It is possible to easily start multiple processes within a
   container, all of which will be managed automatically, with the
   normal `docker run` command.
 - If a managed process dies or crashes, CFEngine will start it again
   within 1 minute.
 - The container itself will live as long as the CFEngine scheduling
   daemon (cf-execd) lives. With CFEngine, we are able to decouple the
   life of the container from the uptime of the service it provides.

## How it works

CFEngine, together with the cfe-docker integration policies, are
installed as part of the Dockerfile. This builds CFEngine into our
Docker image.

The Dockerfile's `ENTRYPOINT` takes an arbitrary
amount of commands (with any desired arguments) as parameters. When we
run the Docker container these parameters get written to CFEngine
policies and CFEngine takes over to ensure that the desired processes
are running in the container.

CFEngine scans the process table for the `basename` of the commands given
to the `ENTRYPOINT` and runs the command to start the process if the `basename`
is not found. For example, if we start the container with
`docker run "/path/to/my/application parameters"`, CFEngine will look for a
process named `application` and run the command. If an entry for `application`
is not found in the process table at any point in time, CFEngine will execute
`/path/to/my/application parameters` to start the application once again. The
check on the process table happens every minute.

Note that it is therefore important that the command to start your
application leaves a process with the basename of the command. This can
be made more flexible by making some minor adjustments to the CFEngine
policies, if desired.

## Usage

This example assumes you have Docker installed and working. We will
install and manage `apache2` and `sshd`
in a single container.

There are three steps:

1. Install CFEngine into the container.
2. Copy the CFEngine Docker process management policy into the
   containerized CFEngine installation.
3. Start your application processes as part of the `docker run` command.

### Building the image

The first two steps can be done as part of a Dockerfile, as follows.

    FROM ubuntu
    MAINTAINER Eystein Måløy Stenberg <eytein.stenberg@gmail.com>

    RUN apt-get update && apt-get install -y wget lsb-release unzip ca-certificates

    # install latest CFEngine
    RUN wget -qO- http://cfengine.com/pub/gpg.key | apt-key add -
    RUN echo "deb http://cfengine.com/pub/apt $(lsb_release -cs) main" > /etc/apt/sources.list.d/cfengine-community.list
    RUN apt-get update && apt-get install -y cfengine-community

    # install cfe-docker process management policy
    RUN wget https://github.com/estenberg/cfe-docker/archive/master.zip -P /tmp/ && unzip /tmp/master.zip -d /tmp/
    RUN cp /tmp/cfe-docker-master/cfengine/bin/* /var/cfengine/bin/
    RUN cp /tmp/cfe-docker-master/cfengine/inputs/* /var/cfengine/inputs/
    RUN rm -rf /tmp/cfe-docker-master /tmp/master.zip

    # apache2 and openssh are just for testing purposes, install your own apps here
    RUN apt-get update && apt-get install -y openssh-server apache2
    RUN mkdir -p /var/run/sshd
    RUN echo "root:password" | chpasswd  # need a password for ssh

    ENTRYPOINT ["/var/cfengine/bin/docker_processes_run.sh"]

By saving this file as Dockerfile to a working directory, you can then build
your image with the docker build command, e.g.,
`docker build -t managed_image`.

### Testing the container

Start the container with `apache2` and `sshd` running and managed, forwarding
a port to our SSH instance:

    $ docker run -p 127.0.0.1:222:22 -d managed_image "/usr/sbin/sshd" "/etc/init.d/apache2 start"

We now clearly see one of the benefits of the cfe-docker integration: it
allows to start several processes as part of a normal `docker run` command.

We can now log in to our new container and see that both `apache2` and `sshd`
are running. We have set the root password to "password" in the Dockerfile
above and can use that to log in with ssh:

    ssh -p222 root@127.0.0.1

    ps -ef
    UID        PID  PPID  C STIME TTY          TIME CMD
    root         1     0  0 07:48 ?        00:00:00 /bin/bash /var/cfengine/bin/docker_processes_run.sh /usr/sbin/sshd /etc/init.d/apache2 start
    root        18     1  0 07:48 ?        00:00:00 /var/cfengine/bin/cf-execd -F
    root        20     1  0 07:48 ?        00:00:00 /usr/sbin/sshd
    root        32     1  0 07:48 ?        00:00:00 /usr/sbin/apache2 -k start
    www-data    34    32  0 07:48 ?        00:00:00 /usr/sbin/apache2 -k start
    www-data    35    32  0 07:48 ?        00:00:00 /usr/sbin/apache2 -k start
    www-data    36    32  0 07:48 ?        00:00:00 /usr/sbin/apache2 -k start
    root        93    20  0 07:48 ?        00:00:00 sshd: root@pts/0
    root       105    93  0 07:48 pts/0    00:00:00 -bash
    root       112   105  0 07:49 pts/0    00:00:00 ps -ef

If we stop apache2, it will be started again within a minute by
CFEngine.

    service apache2 status
     Apache2 is running (pid 32).
    service apache2 stop
             * Stopping web server apache2 ... waiting    [ OK ]
    service apache2 status
     Apache2 is NOT running.
    # ... wait up to 1 minute...
    service apache2 status
     Apache2 is running (pid 173).

## Adapting to your applications

To make sure your applications get managed in the same manner, there are
just two things you need to adjust from the above example:

 - In the Dockerfile used above, install your applications instead of
   `apache2` and `sshd`.
 - When you start the container with `docker run`,
   specify the command line arguments to your applications rather than
   `apache2` and `sshd`.
