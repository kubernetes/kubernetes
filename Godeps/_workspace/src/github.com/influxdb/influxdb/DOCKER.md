# Docker Setup
========================

This document describes how to build and run a minimal InfluxDB container under Docker.   Currently, it has only been tested for local development and assumes that you have a working docker environment.

## Building Image

To build a docker image for InfluxDB from your current checkout, run the following:

```
$ ./build-docker.sh
```

This script uses the `golang:1.5` image to build a fully static binary of `influxd` and then adds it to a minimal `scratch` image.

To build the image using a different version of go:

```
$ GO_VER=1.4.2 ./build-docker.sh
```

Available version can be found [here](https://hub.docker.com/_/golang/).

## Single Node Container

This will start an interactive, single-node, that publishes the containers port `8086` and `8088` to the hosts ports `8086` and `8088` respectively.  This is identical to starting `influxd` manually.

```
$ docker run -it -p 8086:8086 -p 8088:8088 influxdb
```

## Multi-Node Cluster

This will create a simple 3-node cluster.  The data is stored within the container and will be lost when the container is removed.  This is only useful for test clusters.

The `HOST_IP` env variable should be your host IP if running under linux or the virtualbox VM IP if running under OSX.  On OSX, this would be something like: `$(docker-machine ip dev)` or `$(boot2docker ip)` depending on which docker tool you are using.

```
$ export HOST_IP=<your host/VM IP>
$ docker run -it -p 8086:8086 -p 8088:8088 influxdb -hostname $HOST_IP:8088
$ docker run -it -p 8186:8086 -p 8188:8088 influxdb -hostname $HOST_IP:8188 -join $HOST_IP:8088
$ docker run -it -p 8286:8086 -p 8288:8088 influxdb -hostname $HOST_IP:8288 -join $HOST_IP:8088
```

