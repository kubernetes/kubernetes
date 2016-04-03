<!--[metadata]>
+++
title = "images"
description = "The images command description and usage"
keywords = ["list, docker, images"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# images

    Usage: docker images [OPTIONS] [REPOSITORY]

    List images

      -a, --all=false      Show all images (default hides intermediate images)
      --digests=false      Show digests
      -f, --filter=[]      Filter output based on conditions provided
      --help=false         Print usage
      --no-trunc=false     Don't truncate output
      -q, --quiet=false    Only show numeric IDs

The default `docker images` will show all top level
images, their repository and tags, and their virtual size.

Docker images have intermediate layers that increase reusability,
decrease disk usage, and speed up `docker build` by
allowing each step to be cached. These intermediate layers are not shown
by default.

The `VIRTUAL SIZE` is the cumulative space taken up by the image and all
its parent images. This is also the disk space used by the contents of the
Tar file created when you `docker save` an image.

An image will be listed more than once if it has multiple repository names
or tags. This single image (identifiable by its matching `IMAGE ID`)
uses up the `VIRTUAL SIZE` listed only once.

### Listing the most recently created images

    $ docker images
    REPOSITORY                TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
    <none>                    <none>              77af4d6b9913        19 hours ago        1.089 GB
    committ                   latest              b6fa739cedf5        19 hours ago        1.089 GB
    <none>                    <none>              78a85c484f71        19 hours ago        1.089 GB
    docker                    latest              30557a29d5ab        20 hours ago        1.089 GB
    <none>                    <none>              5ed6274db6ce        24 hours ago        1.089 GB
    postgres                  9                   746b819f315e        4 days ago          213.4 MB
    postgres                  9.3                 746b819f315e        4 days ago          213.4 MB
    postgres                  9.3.5               746b819f315e        4 days ago          213.4 MB
    postgres                  latest              746b819f315e        4 days ago          213.4 MB


## Listing the full length image IDs

    $ docker images --no-trunc
    REPOSITORY                    TAG                 IMAGE ID                                                           CREATED             VIRTUAL SIZE
    <none>                        <none>              77af4d6b9913e693e8d0b4b294fa62ade6054e6b2f1ffb617ac955dd63fb0182   19 hours ago        1.089 GB
    committest                    latest              b6fa739cedf5ea12a620a439402b6004d057da800f91c7524b5086a5e4749c9f   19 hours ago        1.089 GB
    <none>                        <none>              78a85c484f71509adeaace20e72e941f6bdd2b25b4c75da8693efd9f61a37921   19 hours ago        1.089 GB
    docker                        latest              30557a29d5abc51e5f1d5b472e79b7e296f595abcf19fe6b9199dbbc809c6ff4   20 hours ago        1.089 GB
    <none>                        <none>              0124422dd9f9cf7ef15c0617cda3931ee68346455441d66ab8bdc5b05e9fdce5   20 hours ago        1.089 GB
    <none>                        <none>              18ad6fad340262ac2a636efd98a6d1f0ea775ae3d45240d3418466495a19a81b   22 hours ago        1.082 GB
    <none>                        <none>              f9f1e26352f0a3ba6a0ff68167559f64f3e21ff7ada60366e2d44a04befd1d3a   23 hours ago        1.089 GB
    tryout                        latest              2629d1fa0b81b222fca63371ca16cbf6a0772d07759ff80e8d1369b926940074   23 hours ago        131.5 MB
    <none>                        <none>              5ed6274db6ceb2397844896966ea239290555e74ef307030ebb01ff91b1914df   24 hours ago        1.089 GB

## Listing image digests

Images that use the v2 or later format have a content-addressable identifier
called a `digest`. As long as the input used to generate the image is
unchanged, the digest value is predictable. To list image digest values, use
the `--digests` flag:

    $ docker images --digests
    REPOSITORY                         TAG                 DIGEST                                                                    IMAGE ID            CREATED             VIRTUAL SIZE
    localhost:5000/test/busybox        <none>              sha256:cbbf2f9a99b47fc460d422812b6a5adff7dfee951d8fa2e4a98caa0382cfbdbf   4986bf8c1536        9 weeks ago         2.43 MB

When pushing or pulling to a 2.0 registry, the `push` or `pull` command
output includes the image digest. You can `pull` using a digest value. You can
also reference by digest in `create`, `run`, and `rmi` commands, as well as the
`FROM` image reference in a Dockerfile.

## Filtering

The filtering flag (`-f` or `--filter`) format is of "key=value". If there is more
than one filter, then pass multiple flags (e.g., `--filter "foo=bar" --filter "bif=baz"`)

The currently supported filters are:

* dangling (boolean - true or false)
* label (`label=<key>` or `label=<key>=<value>`)

##### Untagged images

    $ docker images --filter "dangling=true"

    REPOSITORY          TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
    <none>              <none>              8abc22fbb042        4 weeks ago         0 B
    <none>              <none>              48e5f45168b9        4 weeks ago         2.489 MB
    <none>              <none>              bf747efa0e2f        4 weeks ago         0 B
    <none>              <none>              980fe10e5736        12 weeks ago        101.4 MB
    <none>              <none>              dea752e4e117        12 weeks ago        101.4 MB
    <none>              <none>              511136ea3c5a        8 months ago        0 B

This will display untagged images, that are the leaves of the images tree (not
intermediary layers). These images occur when a new build of an image takes the
`repo:tag` away from the image ID, leaving it untagged. A warning will be issued
if trying to remove an image when a container is presently using it.
By having this flag it allows for batch cleanup.

Ready for use by `docker rmi ...`, like:

    $ docker rmi $(docker images -f "dangling=true" -q)

    8abc22fbb042
    48e5f45168b9
    bf747efa0e2f
    980fe10e5736
    dea752e4e117
    511136ea3c5a

NOTE: Docker will warn you if any containers exist that are using these untagged images.


