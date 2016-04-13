% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-load - Load an image from a tar archive or STDIN

# SYNOPSIS
**docker load**
[**--help**]
[**-i**|**--input**[=*INPUT*]]


# DESCRIPTION

Loads a tarred repository from a file or the standard input stream.
Restores both images and tags.

# OPTIONS
**--help**
  Print usage statement

**-i**, **--input**=""
   Read from a tar archive file, instead of STDIN

# EXAMPLES

    $ docker images
    REPOSITORY          TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
    busybox             latest              769b9341d937        7 weeks ago         2.489 MB
    $ docker load --input fedora.tar
    $ docker images
    REPOSITORY          TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
    busybox             latest              769b9341d937        7 weeks ago         2.489 MB
    fedora              rawhide             0d20aec6529d        7 weeks ago         387 MB
    fedora              20                  58394af37342        7 weeks ago         385.5 MB
    fedora              heisenbug           58394af37342        7 weeks ago         385.5 MB
    fedora              latest              58394af37342        7 weeks ago         385.5 MB

# See also
**docker-save(1)** to save an image(s) to a tar archive (streamed to STDOUT by default).

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
