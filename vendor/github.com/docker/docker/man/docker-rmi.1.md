% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-rmi - Remove one or more images

# SYNOPSIS
**docker rmi**
[**-f**|**--force**[=*false*]]
[**--help**]
[**--no-prune**[=*false*]]
IMAGE [IMAGE...]

# DESCRIPTION

Removes one or more images from the host node. This does not remove images from
a registry. You cannot remove an image of a running container unless you use the
**-f** option. To see all images on a host use the **docker images** command.

# OPTIONS
**-f**, **--force**=*true*|*false*
   Force removal of the image. The default is *false*.

**--help**
  Print usage statement

**--no-prune**=*true*|*false*
   Do not delete untagged parents. The default is *false*.

# EXAMPLES

## Removing an image

Here is an example of removing an image:

    docker rmi fedora/httpd

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
April 2015, updated by Mary Anthony for v2 <mary@docker.com>
