# Docker Registry Integration Testing

These integration tests cover interactions between the Docker daemon and the
registry server. All tests are run using the docker cli.

The compose configuration is intended to setup a testing environment for Docker
using multiple registry configurations. These configurations include different
combinations of a v1 and v2 registry as well as TLS configurations.

## Running inside of Docker
### Get integration container
The container image to run the integation tests will need to be pulled or built
locally.

*Building locally*
```
$ docker build -t distribution/docker-integration .
```

### Run script

Invoke the tests within Docker through the `run.sh` script.

```
$ ./run.sh
```

Run with aufs driver and tmp volume
**NOTE: Using a volume will prevent multiple runs from needing to
re-pull images**
```
$ DOCKER_GRAPHDRIVER=aufs DOCKER_VOLUME=/tmp/volume ./run.sh
```

### Example developer flow 

These tests are useful for developing both as a registry and docker
core developer. The following setup may be used to do integration
testing between development versions

Insert into your `.zshrc` or `.bashrc`

```
# /usr/lib/docker for Docker-in-Docker
# Set this directory to make each invocation run much faster, without
# the need to repull images.
export DOCKER_VOLUME=$HOME/.docker-test-volume

# Use overlay for all Docker testing, try aufs if overlay not supported
export DOCKER_GRAPHDRIVER=overlay

# Name this according to personal preference
function rdtest() {
  if [ "$1" != "" ]; then
    DOCKER_BINARY=$GOPATH/src/github.com/docker/docker/bundles/$1/binary/docker
    if [ ! -f $DOCKER_BINARY ]; then
      current_version=`cat $GOPATH/src/github.com/docker/docker/VERSION`
      echo "$DOCKER_BINARY does not exist"
      echo "Current checked out docker version: $current_version"
      echo "Checkout desired version and run 'make binary' from $GOPATH/src/github.com/docker/docker"
      return 1
    fi
  fi

  $GOPATH/src/github.com/docker/distribution/contrib/docker-integration/run.sh
}
```

Run with Docker release version
```
$ rdtest
```

Run using local development version of docker
```
$ cd $GOPATH/src/github.com/docker/docker
$ make binary
$ rdtest `cat VERSION`
```

## Running manually outside of Docker

### Install Docker Compose

[Docker Compose Installation Guide](https://docs.docker.com/compose/install/)

### Start compose setup
```
docker-compose up
```

### Install Certificates
The certificates must be installed in /etc/docker/cert.d in order to use TLS
client auth and use the CA certificate.
```
sudo sh ./install_certs.sh
```

### Test with Docker
Tag an image as with any other private registry. Attempt to push the image.

```
docker pull hello-world
docker tag hello-world localhost:5440/hello-world
docker push localhost:5440/hello-world

docker tag hello-world localhost:5441/hello-world
docker push localhost:5441/hello-world
# Perform login using user `testuser` and password `passpassword`
```

### Set /etc/hosts entry
Find the non-localhost ip address of local machine

### Run bats
Run the bats tests after updating /etc/hosts, installing the certificates, and
running the `docker-compose` script.
```
bats -p .
```

## Configurations

Port | V2 | TLS | Authentication
--- | --- | --- | ---
5000 | yes | no | none
5002 | yes | no | none
5440 | yes | yes | none
5441 | yes | yes | basic (testuser/passpassword)
5442 | yes | yes | TLS client
5443 | yes | yes | TLS client (no CA)
5444 | yes | yes | TLS client + basic (testuser/passpassword)
5445 | yes | yes (no CA) | none
5446 | yes | yes (no CA) | basic (testuser/passpassword)
5447 | yes | yes (no CA) | TLS client
5448 | yes | yes (SSLv3) | none
