<!--[metadata]>
+++
title = "rm"
description = "The rm command description and usage"
keywords = ["remove, Docker, container"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# rm

    Usage: docker rm [OPTIONS] CONTAINER [CONTAINER...]

    Remove one or more containers

      -f, --force=false      Force the removal of a running container (uses SIGKILL)
      -l, --link=false       Remove the specified link
      -v, --volumes=false    Remove the volumes associated with the container

## Examples

    $ docker rm /redis
    /redis

This will remove the container referenced under the link
`/redis`.

    $ docker rm --link /webapp/redis
    /webapp/redis

This will remove the underlying link between `/webapp` and the `/redis`
containers removing all network communication.

    $ docker rm --force redis
    redis

The main process inside the container referenced under the link `/redis` will receive
`SIGKILL`, then the container will be removed.

    $ docker rm $(docker ps -a -q)

This command will delete all stopped containers. The command 
`docker ps -a -q` will return all existing container IDs and pass them to 
the `rm` command which will delete them. Any running containers will not be
deleted.

