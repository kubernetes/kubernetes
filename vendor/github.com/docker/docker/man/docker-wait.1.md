% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-wait - Block until a container stops, then print its exit code.

# SYNOPSIS
**docker wait**
[**--help**]
CONTAINER [CONTAINER...]

# DESCRIPTION

Block until a container stops, then print its exit code.

# OPTIONS
**--help**
  Print usage statement

# EXAMPLES

    $ docker run -d fedora sleep 99
    079b83f558a2bc52ecad6b2a5de13622d584e6bb1aea058c11b36511e85e7622
    $ docker wait 079b83f558a2bc
    0

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
