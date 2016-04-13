% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-top - Display the running processes of a container

# SYNOPSIS
**docker top**
[**--help**]
CONTAINER [ps OPTIONS]

# DESCRIPTION

Display the running process of the container. ps-OPTION can be any of the
 options you would pass to a Linux ps command.

# OPTIONS
**--help**
  Print usage statement

# EXAMPLES

Run **docker top** with the ps option of -x:

    $ docker top 8601afda2b -x
    PID      TTY       STAT       TIME         COMMAND
    16623    ?         Ss         0:00         sleep 99999


# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
June 2015, updated by Ma Shimiao <mashimiao.fnst@cn.fujitsu.com>
