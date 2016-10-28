## Building and releasing Guestbook Image

This process employs building two docker images, one compiles the source and the other hosts the compiled binaries.

Releasing the image requires that you have access to the docker registry user account which will host the image. You can specify the registry including the user account by setting the environment variable `REGISTRY`.

To build and release the guestbook image:

    cd examples/guestbook-go/_src
    make release

To build and release the guestbook image with a different registry and version:

    VERSION=v4 REGISTRY="docker.io/luebken" make build

If you want to, you can build and push the image step by step:

    make clean
    make build
    make push


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/guestbook-go/_src/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
