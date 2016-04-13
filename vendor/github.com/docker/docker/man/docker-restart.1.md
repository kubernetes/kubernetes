% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-restart - Restart a running container

# SYNOPSIS
**docker restart**
[**--help**]
[**-t**|**--time**[=*10*]]
CONTAINER [CONTAINER...]

# DESCRIPTION
Restart each container listed.

# OPTIONS
**--help**
  Print usage statement

**-t**, **--time**=10
   Number of seconds to try to stop for before killing the container. Once killed it will then be restarted. Default is 10 seconds.

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
