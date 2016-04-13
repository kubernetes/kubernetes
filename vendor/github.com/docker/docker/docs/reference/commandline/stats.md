<!--[metadata]>
+++
title = "stats"
description = "The stats command description and usage"
keywords = ["container, resource, statistics"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# stats

    Usage: docker stats CONTAINER [CONTAINER...]

    Display a live stream of one or more containers' resource usage statistics

      --help=false       Print usage
      --no-stream=false  Disable streaming stats and only pull the first result

Running `docker stats` on multiple containers

    $ docker stats redis1 redis2
    CONTAINER           CPU %               MEM USAGE/LIMIT     MEM %               NET I/O
    redis1              0.07%               796 KB/64 MB        1.21%               788 B/648 B
    redis2              0.07%               2.746 MB/64 MB      4.29%               1.266 KB/648 B


The `docker stats` command will only return a live stream of data for running
containers. Stopped containers will not return any data.

> **Note:**
> If you want more detailed information about a container's resource
> usage, use the API endpoint.