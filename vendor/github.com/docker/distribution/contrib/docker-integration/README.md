# Docker Registry Integration Testing

These integration tests cover interactions between registry clients such as
the docker daemon and the registry server. All tests can be run using the
[golem integration test runner](https://github.com/docker/golem)

The integration tests configure components using docker compose
(see docker-compose.yaml) and the runner can be using the golem
configuration file (see golem.conf).

## Running integration tests

### Run using multiversion script

The integration tests in the `contrib/docker-integration` directory can be simply
run by executing the run script `./run_multiversion.sh`. If there is no running
daemon to connect to, run as `./run_multiversion.sh -d`.

This command will build the distribution image from the locally checked out
version and run against multiple versions of docker defined in the script. To
run a specific version of the registry or docker, Golem will need to be
executed manually.

### Run manually using Golem

Using the golem tool directly allows running against multiple versions of
the registry and docker. Running against multiple versions of the registry
can be useful for testing changes in the docker daemon which are not
covered by the default run script.

#### Installing Golem

Golem is distributed as an executable binary which can be installed from
the [release page](https://github.com/docker/golem/releases/tag/v0.1).

#### Running golem with docker

Additionally golem can be run as a docker image requiring no additonal
installation.

`docker run --privileged -v "$GOPATH/src/github.com/docker/distribution/contrib/docker-integration:/test" -w /test distribution/golem golem -rundaemon .`

#### Golem custom images

Golem tests version of software by defining the docker image to test.

Run with registry 2.2.1 and docker 1.10.3

`golem -i golem-dind:latest,docker:1.10.3-dind,1.10.3 -i golem-distribution:latest,registry:2.2.1 .`


#### Use golem caching for developing tests

Golem allows caching image configuration to reduce test start up time.
Using this cache will allow tests with the same set of images to start
up quickly. This can be useful when developing tests and needing the
test to run quickly. If there are changes which effect the image (such as
building a new registry image), then startup time will be slower.

Run this command multiple times and after the first time test runs
should start much quicker.
`golem -cache ~/.cache/docker/golem -i golem-dind:latest,docker:1.10.3-dind,1.10.3 -i golem-distribution:latest,registry:2.2.1 .`

