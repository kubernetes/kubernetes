## Building and releasing Guestbook Image

Guestbook build process employs the usage of docker-in-docker to build an image within another. This requires that the build image has access to the `docker` program's binary, which defaults to the docker available on your host machine. In the case of boot2docker, `DOCKER_BIN` must be set to the binary's location in the boot2docker's vm.

Releasing the image requires that you have access to the docker registry user account which will host the image.

To build and release the guestbook image:

    cd examples/guestbook-go/src
    ./script/release.sh

If you're using boot2docker, specify the `DOCKER_BIN` environment variable

    DOCKER_BIN="$(boot2docker ssh which docker)" ./script/release.sh

#### Step by step

If you may want to, you can build and push the image step by step.

###### Start fresh before building

    ./script/clean.sh 2> /dev/null

###### Build

Builds a docker image that builds the app and packages it into a minimal docker image

    ./script/build.sh

If you're using boot2docker, specify the `DOCKER_BIN` environment variable

    DOCKER_BIN="$(boot2docker ssh which docker)" ./script/build.sh

###### Push

Accepts an optional tag (defaults to "latest")

    ./script/push.sh [TAG]

###### Clean up

    ./script/clean.sh
