<!--[metadata]>
+++
title = "exec"
description = "The exec command description and usage"
keywords = ["command, container, run, execute"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# exec

    Usage: docker exec [OPTIONS] CONTAINER COMMAND [ARG...]

    Run a command in a running container

      -d, --detach=false         Detached mode: run command in the background
      -i, --interactive=false    Keep STDIN open even if not attached
      -t, --tty=false            Allocate a pseudo-TTY
      -u, --user=                Username or UID (format: <name|uid>[:<group|gid>])

The `docker exec` command runs a new command in a running container.

The command started using `docker exec` only runs while the container's primary
process (`PID 1`) is running, and it is not restarted if the container is
restarted.

If the container is paused, then the `docker exec` command will fail with an error:

    $ docker pause test
    test
    $ docker ps
    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                   PORTS               NAMES
    1ae3b36715d2        ubuntu:latest       "bash"              17 seconds ago      Up 16 seconds (Paused)                       test
    $ docker exec test ls
    FATA[0000] Error response from daemon: Container test is paused, unpause the container before exec
    $ echo $?
    1

## Examples

    $ docker run --name ubuntu_bash --rm -i -t ubuntu bash

This will create a container named `ubuntu_bash` and start a Bash session.

    $ docker exec -d ubuntu_bash touch /tmp/execWorks

This will create a new file `/tmp/execWorks` inside the running container
`ubuntu_bash`, in the background.

    $ docker exec -it ubuntu_bash bash

This will create a new Bash session in the container `ubuntu_bash`.

