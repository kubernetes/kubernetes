<!--[metadata]>
+++
title = "rmi"
description = "The rmi command description and usage"
keywords = ["remove, image, Docker"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# rmi

    Usage: docker rmi [OPTIONS] IMAGE [IMAGE...]

    Remove one or more images

      -f, --force=false    Force removal of the image
      --no-prune=false     Do not delete untagged parents


You can remove an image using its short or long ID, its tag, or its digest. If
an image has one or more tag or digest reference, you must remove all of them
before the image is removed.

    $ docker images
    REPOSITORY                TAG                 IMAGE ID            CREATED             SIZE
    test1                     latest              fd484f19954f        23 seconds ago      7 B (virtual 4.964 MB)
    test                      latest              fd484f19954f        23 seconds ago      7 B (virtual 4.964 MB)
    test2                     latest              fd484f19954f        23 seconds ago      7 B (virtual 4.964 MB)

    $ docker rmi fd484f19954f
    Error: Conflict, cannot delete image fd484f19954f because it is tagged in multiple repositories, use -f to force
    2013/12/11 05:47:16 Error: failed to remove one or more images

    $ docker rmi test1
    Untagged: test1:latest
    $ docker rmi test2
    Untagged: test2:latest

    $ docker images
    REPOSITORY                TAG                 IMAGE ID            CREATED             SIZE
    test                      latest              fd484f19954f        23 seconds ago      7 B (virtual 4.964 MB)
    $ docker rmi test
    Untagged: test:latest
    Deleted: fd484f19954f4920da7ff372b5067f5b7ddb2fd3830cecd17b96ea9e286ba5b8

If you use the `-f` flag and specify the image's short or long ID, then this
command untags and removes all images that match the specified ID.

    $ docker images
    REPOSITORY                TAG                 IMAGE ID            CREATED             SIZE
    test1                     latest              fd484f19954f        23 seconds ago      7 B (virtual 4.964 MB)
    test                      latest              fd484f19954f        23 seconds ago      7 B (virtual 4.964 MB)
    test2                     latest              fd484f19954f        23 seconds ago      7 B (virtual 4.964 MB)

    $ docker rmi -f fd484f19954f
    Untagged: test1:latest
    Untagged: test:latest
    Untagged: test2:latest
    Deleted: fd484f19954f4920da7ff372b5067f5b7ddb2fd3830cecd17b96ea9e286ba5b8

An image pulled by digest has no tag associated with it:

    $ docker images --digests
    REPOSITORY                     TAG       DIGEST                                                                    IMAGE ID        CREATED         VIRTUAL SIZE
    localhost:5000/test/busybox    <none>    sha256:cbbf2f9a99b47fc460d422812b6a5adff7dfee951d8fa2e4a98caa0382cfbdbf   4986bf8c1536    9 weeks ago     2.43 MB

To remove an image using its digest:

    $ docker rmi localhost:5000/test/busybox@sha256:cbbf2f9a99b47fc460d422812b6a5adff7dfee951d8fa2e4a98caa0382cfbdbf
    Untagged: localhost:5000/test/busybox@sha256:cbbf2f9a99b47fc460d422812b6a5adff7dfee951d8fa2e4a98caa0382cfbdbf
    Deleted: 4986bf8c15363d1c5d15512d5266f8777bfba4974ac56e3270e7760f6f0a8125
    Deleted: ea13149945cb6b1e746bf28032f02e9b5a793523481a0a18645fc77ad53c4ea2
    Deleted: df7546f9f060a2268024c8a230d8639878585defcc1bc6f79d2728a13957871b

