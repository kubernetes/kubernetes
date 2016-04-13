% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-tag - Tag an image into a repository

# SYNOPSIS
**docker tag**
[**-f**|**--force**[=*false*]]
[**--help**]
IMAGE[:TAG] [REGISTRY_HOST/][USERNAME/]NAME[:TAG]

# DESCRIPTION
Assigns a new alias to an image in a registry. An alias refers to the
entire image name including the optional `TAG` after the ':'. 

If you do not specify a `REGISTRY_HOST`, the command uses Docker's public
registry located at `registry-1.docker.io` by default. 

# "OPTIONS"
**-f**, **--force**=*true*|*false*
   When set to true, force the alias. The default is *false*.

**REGISTRYHOST**
   The hostname of the registry if required. This may also include the port
separated by a ':'

**USERNAME**
   The username or other qualifying identifier for the image.

**NAME**
   The image name.

**TAG**
   The tag you are assigning to the image.  Though this is arbitrary it is
recommended to be used for a version to distinguish images with the same name.
Also, for consistency tags should only include a-z0-9-_. .
Note that here TAG is a part of the overall name or "tag".

# OPTIONS
**-f**, **--force**=*true*|*false*
   Force. The default is *false*.

# EXAMPLES

## Giving an image a new alias

Here is an example of aliasing an image (e.g., 0e5574283393) as "httpd" and 
tagging it into the "fedora" repository with "version1.0":

    docker tag 0e5574283393 fedora/httpd:version1.0

## Tagging an image for a private repository

To push an image to an private registry and not the central Docker
registry you must tag it with the registry hostname and port (if needed).

    docker tag 0e5574283393 myregistryhost:5000/fedora/httpd:version1.0

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
July 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
April 2015, updated by Mary Anthony for v2 <mary@docker.com>
June 2015, updated by Sally O'Malley <somalley@redhat.com>
