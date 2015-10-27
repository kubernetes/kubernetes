<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Building and releasing Guestbook Image

This process employs building two docker images, one compiles the source and the other hosts the compiled binaries.

Releasing the image requires that you have access to the docker registry user account which will host the image.

To build and release the guestbook image:

    cd examples/guestbook-go/_src
    ./script/release.sh

#### Step by step

If you may want to, you can build and push the image step by step.

###### Start fresh before building

    ./script/clean.sh 2> /dev/null

###### Build

Builds a docker image that builds the app and packages it into a minimal docker image

    ./script/build.sh

###### Push

Accepts an optional tag (defaults to "latest")

    ./script/push.sh [TAG]

###### Clean up

    ./script/clean.sh




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/guestbook-go/_src/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
