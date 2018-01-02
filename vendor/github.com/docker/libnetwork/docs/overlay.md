# Overlay Driver

### Design
TODO

### Multi-Host Overlay Driver Quick Start

This example is to provision two Docker Hosts with the **experimental** Libnetwork overlay network driver.

### Pre-Requisites

- Kernel >= 3.16
- Experimental Docker client

### Install Docker Experimental

Follow Docker experimental installation instructions at: [https://github.com/docker/docker/tree/master/experimental](https://github.com/docker/docker/tree/master/experimental)

To ensure you are running the experimental Docker branch, check the version and look for the experimental tag:

```
$ docker -v
Docker version 1.8.0-dev, build f39b9a0, experimental
```

### Install and Bootstrap K/V Store


Multi-host networking uses a pluggable Key-Value store backend to distribute states using `libkv`.
`libkv` supports multiple pluggable backends such as `consul`, `etcd` & `zookeeper` (more to come).

In this example we will use `consul`

Install:

```
$ curl -OL https://dl.bintray.com/mitchellh/consul/0.5.2_linux_amd64.zip
$ unzip 0.5.2_linux_amd64.zip
$ mv consul /usr/local/bin/
```

**host-1** Start Consul as a server in bootstrap mode:

``` 
$ consul agent -server -bootstrap -data-dir /tmp/consul -bind=<host-1-ip-address>
```

**host-2** Start the Consul agent:

``` 
$ consul agent -data-dir /tmp/consul -bind=<host-2-ip-address>
$ consul join <host-1-ip-address>
```


### Start the Docker Daemon with the Network Driver Daemon Flags

**host-1** Docker daemon:

```
$ docker -d --kv-store=consul:localhost:8500 --label=com.docker.network.driver.overlay.bind_interface=eth0
```

**host-2** Start the Docker Daemon with the neighbor ID configuration:

```
$ docker -d --kv-store=consul:localhost:8500 --label=com.docker.network.driver.overlay.bind_interface=eth0 --label=com.docker.network.driver.overlay.neighbor_ip=<host-1-ip-address>
```

### QuickStart Containers Attached to a Network

**host-1** Start a container that publishes a service svc1 in the network dev that is managed by overlay driver.

```
$ docker run -i -t --publish-service=svc1.dev.overlay debian
root@21578ff721a9:/# ip add show eth0
34: eth0: <BROADCAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default
    link/ether 02:42:ec:41:35:bf brd ff:ff:ff:ff:ff:ff
    inet 172.21.0.16/16 scope global eth0
       valid_lft forever preferred_lft forever
    inet6 fe80::42:ecff:fe41:35bf/64 scope link
       valid_lft forever preferred_lft forever
```

**host-2** Start a container that publishes a service svc2 in the network dev that is managed by overlay driver.

```
$ docker run -i -t --publish-service=svc2.dev.overlay debian
root@d217828eb876:/# ping svc1
PING svc1 (172.21.0.16): 56 data bytes
64 bytes from 172.21.0.16: icmp_seq=0 ttl=64 time=0.706 ms
64 bytes from 172.21.0.16: icmp_seq=1 ttl=64 time=0.687 ms
64 bytes from 172.21.0.16: icmp_seq=2 ttl=64 time=0.841 ms
```
### Detailed Setup

You can also setup networks and services and then attach a running container to them.

**host-1**:

```
docker network create -d overlay prod 
docker network ls
docker network info prod
docker service publish db1.prod
cid=$(docker run -itd -p 8000:8000 ubuntu)
docker service attach $cid db1.prod
```

**host-2**:

```
docker network ls
docker network info prod
docker service publish db2.prod
cid=$(docker run -itd -p 8000:8000 ubuntu)
docker service attach $cid db2.prod
```

Once a container is started, a container on `host-1` and `host-2` both containers should be able to ping one another via IP, service name, \<service name>.\<network name>


View information about the networks and services using `ls` and `info` subcommands like so:

```
$ docker service ls
SERVICE ID          NAME                  NETWORK             CONTAINER
0771deb5f84b        db2                   prod                0e54a527f22c
aea23b224acf        db1                   prod                4b0a309ca311

$ docker network info prod
Network Id: 5ac68be2518959b48ad102e9ec3d8f42fb2ec72056aa9592eb5abd0252203012
	Name: prod
	Type: overlay

$ docker service info db1.prod
Service Id: aea23b224acfd2da9b893870e0d632499188a1a4b3881515ba042928a9d3f465
	Name: db1
	Network: prod
```

To detach and unpublish a service:

```
$ docker service detach $cid <service>.<network>
$ docker service unpublish <service>.<network>

# Example:
$ docker service detach $cid  db2.prod
$ docker service unpublish db2.prod
```

To reiterate, this is experimental, and will be under active development.
