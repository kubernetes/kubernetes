Bridge Driver
=============

The bridge driver is an implementation that uses Linux Bridging and iptables to provide connectivity for containers
It creates a single bridge, called `docker0` by default, and attaches a `veth pair` between the bridge and every endpoint.

## Configuration

The bridge driver supports configuration through the Docker Daemon flags. 

## Usage

This driver is supported for the default "bridge" network only and it cannot be used for any other networks.
