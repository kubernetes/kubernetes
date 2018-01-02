# Docker Setup
========================

This document describes how to build and run a minimal InfluxDB container under Docker.   Currently, it has only been tested for local development and assumes that you have a working docker environment.

## Building Image

To build a docker image for InfluxDB from your current checkout, run the following:

```
$ ./build-docker.sh
```

This script uses the `golang:1.7.4` image to build a fully static binary of `influxd` and then adds it to a minimal `scratch` image.

To build the image using a different version of go:

```
$ GO_VER=1.7.4 ./build-docker.sh
```

Available version can be found [here](https://hub.docker.com/_/golang/).

## Single Node Container

This will start an interactive, single-node, that publishes the containers port `8086` and `8088` to the hosts ports `8086` and `8088` respectively.  This is identical to starting `influxd` manually.

```
$ docker run -it -p 8086:8086 -p 8088:8088 influxdb
```
