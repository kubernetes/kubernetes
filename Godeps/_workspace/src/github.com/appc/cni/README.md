# cni - the Container Network Interface

## What is CNI?

CNI, the _Container Network Interface_, is a proposed standard for configuring network interfaces for Linux application containers.
The standard consists of a simple specification for how executable plugins can be used to configure network namespaces, and a go library implementing that specification.

The specification itself is contained in [SPEC.md](SPEC.md)

## Why develop CNI?

Application containers on Linux are a rapidly evolving area, and within this space networking is a particularly unsolved problem, as it is highly environment-specific.
We believe that every container runtime will seek to solve the same problem of making the network layer pluggable.
In order to avoid duplication, we think it is prudent to define a common interface between the network plugins and container execution.
Hence we are proposing this specification, along with an initial set of plugins that can be used by different container runtime systems.

## How do I use CNI?

## Requirements
CNI requires Go 1.4+ to build.

## Included Plugins
This repository includes a number of common plugins that can be found in plugins/ directory.
Please see Documentation/ folder for documentation about particular plugins.

## Running the plugins
The scripts/ directory contains two scripts, priv-net-run.sh and docker-run.sh, that can be used to excercise the plugins.

Start out by creating a netconf file to describe a network:

```
$ mkdir -p /etc/cni/net.d
$ cat >/etc/cni/net.d/10-mynet.conf <<EOF
{
	"name": "mynet",
	"type": "bridge",
	"bridge": "cni0",
	"isGateway": true,
	"ipMasq": true,
	"ipam": {
		"type": "host-local",
		"subnet": "10.22.0.0/16",
		"routes": [
			{ "dst": "0.0.0.0/0" }
		]
	}
}
EOF
```

The directory `/etc/cni/net.d` is the default location in which the scripts will look for net configurations.

Next, build the plugins:

```
$ ./build
```

Finally, execute a command (`ifconfig` in this example) in a private network namespace that has joined `mynet` network:

```
$ CNI_PATH=`pwd`/bin
$ cd scripts
$ sudo CNI_PATH=$CNI_PATH ./priv-net-run.sh ifconfig
eth0      Link encap:Ethernet  HWaddr f2:c2:6f:54:b8:2b  
          inet addr:10.22.0.2  Bcast:0.0.0.0  Mask:255.255.0.0
          inet6 addr: fe80::f0c2:6fff:fe54:b82b/64 Scope:Link
          UP BROADCAST MULTICAST  MTU:1500  Metric:1
          RX packets:1 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:1 overruns:0 carrier:0
          collisions:0 txqueuelen:0 
          RX bytes:90 (90.0 B)  TX bytes:0 (0.0 B)

lo        Link encap:Local Loopback  
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0 
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)
```

The environment variable `CNI_PATH` tells the scripts and library where to look for plugins executables.

## Running a Docker container with network namespace set up by CNI plugins

Use instructions in the previous section to define a netconf and build the plugins.
Next, docker-run.sh script wraps `docker run` command to execute the plugins prior to entering the container:

```
$ CNI_PATH=`pwd`/bin
$ cd scripts
$ sudo CNI_PATH=$CNI_PATH ./docker-run.sh --rm busybox:latest /sbin/ifconfig
eth0      Link encap:Ethernet  HWaddr fa:60:70:aa:07:d1  
          inet addr:10.22.0.2  Bcast:0.0.0.0  Mask:255.255.0.0
          inet6 addr: fe80::f860:70ff:feaa:7d1/64 Scope:Link
          UP BROADCAST MULTICAST  MTU:1500  Metric:1
          RX packets:1 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:1 overruns:0 carrier:0
          collisions:0 txqueuelen:0 
          RX bytes:90 (90.0 B)  TX bytes:0 (0.0 B)

lo        Link encap:Local Loopback  
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0 
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)
```
