<!--[metadata]>
+++
title = "port"
description = "The port command description and usage"
keywords = ["port, mapping, container"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# port

    Usage: docker port CONTAINER [PRIVATE_PORT[/PROTO]]

    List port mappings for the CONTAINER, or lookup the public-facing port that is
	NAT-ed to the PRIVATE_PORT

You can find out all the ports mapped by not specifying a `PRIVATE_PORT`, or
just a specific mapping:

    $ docker ps test
    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                                            NAMES
    b650456536c7        busybox:latest      top                 54 minutes ago      Up 54 minutes       0.0.0.0:1234->9876/tcp, 0.0.0.0:4321->7890/tcp   test
    $ docker port test
    7890/tcp -> 0.0.0.0:4321
    9876/tcp -> 0.0.0.0:1234
    $ docker port test 7890/tcp
    0.0.0.0:4321
    $ docker port test 7890/udp
    2014/06/24 11:53:36 Error: No public port '7890/udp' published for test
    $ docker port test 7890
    0.0.0.0:4321
