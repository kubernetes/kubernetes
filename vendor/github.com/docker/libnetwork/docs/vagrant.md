# Vagrant Setup to Test the Overlay Driver

This documentation highlights how to use Vagrant to start a three nodes setup to test Docker network.

## Pre-requisites

This was tested on:

- Vagrant 1.7.2
- VirtualBox 4.3.26

## Machine Setup

The Vagrantfile provided will start three virtual machines. One will act as a consul server, and the other two will act as Docker host.
The experimental version of Docker is installed.

- `consul-server` is the Consul server node, based on Ubuntu 14.04, this has IP 192.168.33.10
- `net-1` is the first Docker host based on Ubuntu 14.10, this has IP 192.168.33.11
- `net-2` is the second Docker host based on Ubuntu 14.10, this has IP 192.168.33.12

## Getting Started

Clone this repo, change to the `docs` directory and let Vagrant do the work.

    $ vagrant up
    $ vagrant status
    Current machine states:

    consul-server             running (virtualbox)
    net-1                     running (virtualbox)
    net-2                     running (virtualbox)

You are now ready to SSH to the Docker hosts and start containers.

    $ vagrant ssh net-1
    vagrant@net-1:~$ docker version
    Client version: 1.8.0-dev
    ...<snip>...

Check that Docker network is functional by listing the default networks:

    vagrant@net-1:~$ docker network ls
    NETWORK ID          NAME                TYPE
    4275f8b3a821        none                null                
    80eba28ed4a7        host                host                
    64322973b4aa        bridge              bridge              

No services has been published so far, so the `docker service ls` will return an empty list:

    $ docker service ls
    SERVICE ID          NAME                NETWORK             CONTAINER

Start a container and check the content of `/etc/hosts`.

    $ docker run -it --rm ubuntu:14.04 bash
    root@df479e660658:/# cat /etc/hosts
    172.21.0.3	df479e660658
    127.0.0.1	localhost
    ::1	localhost ip6-localhost ip6-loopback
    fe00::0	ip6-localnet
    ff00::0	ip6-mcastprefix
    ff02::1	ip6-allnodes
    ff02::2	ip6-allrouters
    172.21.0.3	distracted_bohr
    172.21.0.3	distracted_bohr.multihost

In a separate terminal on `net-1` list the networks again. You will see that the _multihost_ overlay now appears.
The overlay network _multihost_ is your default network. This was setup by the Docker daemon during the Vagrant provisioning. Check `/etc/default/docker` to see the options that were set.

    vagrant@net-1:~$ docker network ls
    NETWORK ID          NAME                TYPE
    4275f8b3a821        none                null
    80eba28ed4a7        host                host
    64322973b4aa        bridge              bridge
    b5c9f05f1f8f        multihost           overlay

Now in a separate terminal, SSH to `net-2`, check the network and services. The networks will be the same, and the default network will also be _multihost_ of type overlay. But the service will show the container started on `net-1`:

    $ vagrant ssh net-2
    vagrant@net-2:~$ docker service ls
    SERVICE ID          NAME                NETWORK             CONTAINER
    b00f2bfd81ac        distracted_bohr     multihost           df479e660658

Start a container on `net-2` and check the `/etc/hosts`.

    vagrant@net-2:~$ docker run -ti --rm ubuntu:14.04 bash
    root@2ac726b4ce60:/# cat /etc/hosts
    172.21.0.4	2ac726b4ce60
    127.0.0.1	localhost
    ::1	localhost ip6-localhost ip6-loopback
    fe00::0	ip6-localnet
    ff00::0	ip6-mcastprefix
    ff02::1	ip6-allnodes
    ff02::2	ip6-allrouters
    172.21.0.3	distracted_bohr
    172.21.0.3	distracted_bohr.multihost
    172.21.0.4	modest_curie
    172.21.0.4	modest_curie.multihost

You will see not only the container that you just started on `net-2` but also the container that you started earlier on `net-1`.
And of course you will be able to ping each container.

## Creating a Non Default Overlay Network

In the previous test we started containers with regular options `-ti --rm` and these containers got placed automatically in the default network which was set to be the _multihost_ network of type overlay.

But you could create your own overlay network and start containers in it. Let's create a new overlay network.
On one of your Docker hosts, `net-1` or `net-2` do:

    $ docker network create -d overlay foobar
    8805e22ad6e29cd7abb95597c91420fdcac54f33fcdd6fbca6dd4ec9710dd6a4
    $ docker network ls
    NETWORK ID          NAME                TYPE
    a77e16a1e394        host                host                
    684a4bb4c471        bridge              bridge              
    8805e22ad6e2        foobar              overlay             
    b5c9f05f1f8f        multihost           overlay             
    67d5a33a2e54        none                null   

Automatically, the second host will also see this network. To start a container on this new network, simply use the `--publish-service` option of `docker run` like so:

    $ docker run -it --rm --publish-service=bar.foobar.overlay ubuntu:14.04 bash

Note, that you could directly start a container with a new overlay using the `--publish-service` option and it will create the network automatically.

Check the docker services now:

    $ docker service ls
    SERVICE ID          NAME                NETWORK             CONTAINER
    b1ffdbfb1ac6        bar                 foobar              6635a3822135

Repeat the getting started steps, by starting another container in this new overlay on the other host, check the `/etc/hosts` file and try to ping each container.

## A look at the interfaces

This new Docker multihost networking is made possible via VXLAN tunnels and the use of network namespaces.
Check the [design](design.md) documentation for all the details. But to explore these concepts a bit, nothing beats an example.

With a running container in one overlay, check the network namespace:

    $ docker inspect -f '{{ .NetworkSettings.SandboxKey}}' 6635a3822135
    /var/run/docker/netns/6635a3822135

This is a none default location for network namespaces which might confuse things a bit. So let's become root, head over to this directory that contains the network namespaces of the containers and check the interfaces:

    $ sudo su
    root@net-2:/home/vagrant# cd /var/run/docker/
    root@net-2:/var/run/docker# ls netns
    6635a3822135
    8805e22ad6e2

To be able to check the interfaces in those network namespace using `ip` command, just create a symlink for `netns` that points to `/var/run/docker/netns`:

    root@net-2:/var/run# ln -s /var/run/docker/netns netns
    root@net-2:/var/run# ip netns show
    6635a3822135
    8805e22ad6e2

The two namespace ID return are the ones of the running container on that host and the one of the actual overlay network the container is in.
Let's check the interfaces in the container:

    root@net-2:/var/run/docker# ip netns exec 6635a3822135 ip addr show eth0
    15: eth0: <BROADCAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default 
        link/ether 02:42:b3:91:22:c3 brd ff:ff:ff:ff:ff:ff
        inet 172.21.0.5/16 scope global eth0
           valid_lft forever preferred_lft forever
        inet6 fe80::42:b3ff:fe91:22c3/64 scope link 
           valid_lft forever preferred_lft forever

Indeed we get back the network interface of our running container, same MAC address, same IP.
If we check the links of the overlay namespace we see our vxlan interface and the VLAN ID being used.

    root@net-2:/var/run/docker# ip netns exec 8805e22ad6e2 ip -d link show
    ...<snip>...
    14: vxlan1: <BROADCAST,UP,LOWER_UP> mtu 1500 qdisc noqueue master br0 state UNKNOWN mode DEFAULT group default 
        link/ether 7a:af:20:ee:e3:81 brd ff:ff:ff:ff:ff:ff promiscuity 1 
        vxlan id 256 srcport 32768 61000 dstport 8472 proxy l2miss l3miss ageing 300 
        bridge_slave 
    16: veth2: <BROADCAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast master br0 state UP mode DEFAULT group default qlen 1000
        link/ether 46:b1:e2:5c:48:a8 brd ff:ff:ff:ff:ff:ff promiscuity 1 
        veth 
        bridge_slave  

If you sniff packets on these interfaces you will see the traffic between your containers.

