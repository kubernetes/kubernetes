% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-port - List port mappings for the CONTAINER, or lookup the public-facing port that is NAT-ed to the PRIVATE_PORT

# SYNOPSIS
**docker port**
[**--help**]
CONTAINER [PRIVATE_PORT[/PROTO]]

# DESCRIPTION
List port mappings for the CONTAINER, or lookup the public-facing port that is NAT-ed to the PRIVATE_PORT

# OPTIONS
**--help**
  Print usage statement

# EXAMPLES

    # docker ps
    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                                            NAMES
    b650456536c7        busybox:latest      top                 54 minutes ago      Up 54 minutes       0.0.0.0:1234->9876/tcp, 0.0.0.0:4321->7890/tcp   test

## Find out all the ports mapped

    # docker port test
    7890/tcp -> 0.0.0.0:4321
    9876/tcp -> 0.0.0.0:1234

## Find out a specific mapping

    # docker port test 7890/tcp
    0.0.0.0:4321

    # docker port test 7890
    0.0.0.0:4321

## An example showing error for non-existent mapping

    # docker port test 7890/udp
    2014/06/24 11:53:36 Error: No public port '7890/udp' published for test

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
November 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
