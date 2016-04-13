<!--[metadata]>
+++
title = "info"
description = "The info command description and usage"
keywords = ["display, docker, information"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# info


    Usage: docker info

    Display system-wide information

For example:

    $ docker -D info
    Containers: 14
    Images: 52
    Storage Driver: aufs
     Root Dir: /var/lib/docker/aufs
     Backing Filesystem: extfs
     Dirs: 545
    Execution Driver: native-0.2
    Logging Driver: json-file
    Kernel Version: 3.13.0-24-generic
    Operating System: Ubuntu 14.04 LTS
    CPUs: 1
    Name: prod-server-42
    ID: 7TRN:IPZB:QYBB:VPBQ:UMPP:KARE:6ZNR:XE6T:7EWV:PKF4:ZOJD:TPYS
    Total Memory: 2 GiB
    Debug mode (server): false
    Debug mode (client): true
    File Descriptors: 10
    Goroutines: 9
    System Time: Tue Mar 10 18:38:57 UTC 2015
    EventsListeners: 0
    Init Path: /usr/bin/docker
    Docker Root Dir: /var/lib/docker
    Http Proxy: http://test:test@localhost:8080
    Https Proxy: https://test:test@localhost:8080
    No Proxy: 9.81.1.160
    Username: svendowideit
    Registry: [https://index.docker.io/v1/]
    Labels:
     storage=ssd

The global `-D` option tells all `docker` commands to output debug information.

When sending issue reports, please use `docker version` and `docker -D info` to
ensure we know how your setup is configured.


