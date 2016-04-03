<!--[metadata]>
+++
title = "history"
description = "The history command description and usage"
keywords = ["docker, image, history"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# history

    Usage: docker history [OPTIONS] IMAGE

    Show the history of an image

      -H, --human=true     Print sizes and dates in human readable format
      --no-trunc=false     Don't truncate output
      -q, --quiet=false    Only show numeric IDs

To see how the `docker:latest` image was built:

    $ docker history docker
    IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
    3e23a5875458        8 days ago          /bin/sh -c #(nop) ENV LC_ALL=C.UTF-8            0 B
    8578938dd170        8 days ago          /bin/sh -c dpkg-reconfigure locales &&    loc   1.245 MB
    be51b77efb42        8 days ago          /bin/sh -c apt-get update && apt-get install    338.3 MB
    4b137612be55        6 weeks ago         /bin/sh -c #(nop) ADD jessie.tar.xz in /        121 MB
    750d58736b4b        6 weeks ago         /bin/sh -c #(nop) MAINTAINER Tianon Gravi <ad   0 B
    511136ea3c5a        9 months ago                                                        0 B                 Imported from -

To see how the `docker:apache` image was added to a container's base image:

    $ docker history docker:scm
    IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
    2ac9d1098bf1        3 months ago        /bin/bash                                       241.4 MB            Added Apache to Fedora base image
    88b42ffd1f7c        5 months ago        /bin/sh -c #(nop) ADD file:1fd8d7f9f6557cafc7   373.7 MB
    c69cab00d6ef        5 months ago        /bin/sh -c #(nop) MAINTAINER Lokesh Mandvekar   0 B
    511136ea3c5a        19 months ago                                                       0 B                 Imported from -


