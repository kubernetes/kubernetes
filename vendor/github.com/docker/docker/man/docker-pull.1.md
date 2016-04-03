% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-pull - Pull an image or a repository from a registry

# SYNOPSIS
**docker pull**
[**-a**|**--all-tags**[=*false*]]
[**--help**] 
NAME[:TAG] | [REGISTRY_HOST[:REGISTRY_PORT]/]NAME[:TAG]

# DESCRIPTION

This command pulls down an image or a repository from a registry. If
there is more than one image for a repository (e.g., fedora) then all
images for that repository name are pulled down including any tags.

If you do not specify a `REGISTRY_HOST`, the command uses Docker's public
registry located at `registry-1.docker.io` by default. 

# OPTIONS
**-a**, **--all-tags**=*true*|*false*
   Download all tagged images in the repository. The default is *false*.

**--help**
  Print usage statement

# EXAMPLE

# Pull a repository with multiple images
# Note that if the  image is previously downloaded then the status would be
# 'Status: Image is up to date for fedora'

    $ docker pull fedora
    Pulling repository fedora
    ad57ef8d78d7: Download complete
    105182bb5e8b: Download complete
    511136ea3c5a: Download complete
    73bd853d2ea5: Download complete

    Status: Downloaded newer image for fedora

    $ docker images
    REPOSITORY   TAG         IMAGE ID        CREATED      VIRTUAL SIZE
    fedora       rawhide     ad57ef8d78d7    5 days ago   359.3 MB
    fedora       20          105182bb5e8b    5 days ago   372.7 MB
    fedora       heisenbug   105182bb5e8b    5 days ago   372.7 MB
    fedora       latest      105182bb5e8b    5 days ago   372.7 MB

# Pull an image, manually specifying path to Docker's public registry and tag
# Note that if the  image is previously downloaded then the status would be
# 'Status: Image is up to date for registry.hub.docker.com/fedora:20'

    $ docker pull registry.hub.docker.com/fedora:20
    Pulling repository fedora
    3f2fed40e4b0: Download complete 
    511136ea3c5a: Download complete 
    fd241224e9cf: Download complete 

    Status: Downloaded newer image for registry.hub.docker.com/fedora:20

    $ docker images
    REPOSITORY   TAG         IMAGE ID        CREATED      VIRTUAL SIZE
    fedora       20          3f2fed40e4b0    4 days ago   372.7 MB


# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
August 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
April 2015, updated by John Willis <john.willis@docker.com>
April 2015, updated by Mary Anthony for v2 <mary@docker.com>
