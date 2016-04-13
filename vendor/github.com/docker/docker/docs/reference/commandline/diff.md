<!--[metadata]>
+++
title = "diff"
description = "The diff command description and usage"
keywords = ["list, changed, files, container"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# diff

    Usage: docker diff CONTAINER

    Inspect changes on a container's filesystem

List the changed files and directories in a container᾿s filesystem
 There are 3 events that are listed in the `diff`:

1. `A` - Add
2. `D` - Delete
3. `C` - Change

For example:

    $ docker diff 7bb0e258aefe

    C /dev
    A /dev/kmsg
    C /etc
    A /etc/mtab
    A /go
    A /go/src
    A /go/src/github.com
    A /go/src/github.com/docker
    A /go/src/github.com/docker/docker
    A /go/src/github.com/docker/docker/.git
    ....
