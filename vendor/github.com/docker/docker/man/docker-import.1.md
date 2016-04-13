% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2014
# NAME
docker-import - Create an empty filesystem image and import the contents of the tarball (.tar, .tar.gz, .tgz, .bzip, .tar.xz, .txz) into it, then optionally tag it.

# SYNOPSIS
**docker import**
[**-c**|**--change**[= []**]]
[**--help**]
file|URL|- [REPOSITORY[:TAG]]

# OPTIONS
**-c**, **--change**=[]
   Apply specified Dockerfile instructions while importing the image
   Supported Dockerfile instructions: `CMD`|`ENTRYPOINT`|`ENV`|`EXPOSE`|`ONBUILD`|`USER`|`VOLUME`|`WORKDIR`

# DESCRIPTION
Create a new filesystem image from the contents of a tarball (`.tar`,
`.tar.gz`, `.tgz`, `.bzip`, `.tar.xz`, `.txz`) into it, then optionally tag it.

# OPTIONS
**--help**
  Print usage statement

# EXAMPLES

## Import from a remote location

    # docker import http://example.com/exampleimage.tgz example/imagerepo

## Import from a local file

Import to docker via pipe and stdin:

    # cat exampleimage.tgz | docker import - example/imagelocal

Import to a Docker image from a local file.

    # docker import /path/to/exampleimage.tgz 


## Import from a local file and tag

Import to docker via pipe and stdin:

    # cat exampleimageV2.tgz | docker import - example/imagelocal:V-2.0

## Import from a local directory

    # tar -c . | docker import - exampleimagedir

## Apply specified Dockerfile instructions while importing the image
This example sets the docker image ENV variable DEBUG to true by default.

    # tar -c . | docker import -c="ENV DEBUG true" - exampleimagedir

# See also
**docker-export(1)** to export the contents of a filesystem as a tar archive to STDOUT.

# HISTORY
April 2014, Originally compiled by William Henry (whenry at redhat dot com)
based on docker.com source material and internal work.
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
