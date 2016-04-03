% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-kill - Kill a running container using SIGKILL or a specified signal

# SYNOPSIS
**docker kill**
[**--help**]
[**-s**|**--signal**[=*"KILL"*]]
CONTAINER [CONTAINER...]

# DESCRIPTION

The main process inside each container specified will be sent SIGKILL,
 or any signal specified with option --signal.

# OPTIONS
**--help**
  Print usage statement

**-s**, **--signal**="KILL"
   Signal to send to the container

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
 based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
