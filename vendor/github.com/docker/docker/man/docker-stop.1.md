% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-stop - Stop a running container by sending SIGTERM and then SIGKILL after a grace period

# SYNOPSIS
**docker stop**
[**--help**]
[**-t**|**--time**[=*10*]]
CONTAINER [CONTAINER...]

# DESCRIPTION
Stop a running container (Send SIGTERM, and then SIGKILL after
 grace period)

# OPTIONS
**--help**
  Print usage statement

**-t**, **--time**=10
   Number of seconds to wait for the container to stop before killing it. Default is 10 seconds.

#See also
**docker-start(1)** to restart a stopped container.

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
