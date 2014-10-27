# Network Setup Guide

This guide demostrates a network setup that will work for environments with access to layer 2 networking 
(bare metal, vmware, etc). The following steps are not required when following the [Installation Guide](coreos_cloud_config.md).

Please note: With some hypervisors, you may have to enable special settings on the virtual network cards for bridging to work (for example, you need to allow 'MAC address spoofing' in Microsoft Hyper-V).

## Hostnames

On each node ensure the hostname is set.

```
hostnamectl set-hostname master
hostnamectl set-hostname node1
hostnamectl set-hostname node2
```

### Setup /etc/hosts

On each node add the following lines to /etc/hosts:

```
192.168.12.10 master
192.168.12.11 node1
192.168.12.12 node2
```

## Create the cbr0 bridge

On each node run the following commands to setup the cbr0 bridge used by Docker and Kubernetes.

```
brctl addbr cbr0
brctl addif cbr0 ens34
ip link set dev cbr0 mtu 1460
ip addr add 10.244.0.1/24 dev cbr0   # this will be different for each minion
ip link set dev cbr0 up
ip route add 10.0.0.0/8 dev cbr0
```

Each node should use a different address. For example:

master

```
ip addr add 10.244.0.1/24 dev cbr0
```

node1

```
ip addr add 10.244.1.1/24 dev cbr0
```

node2

```
ip addr add 10.244.2.1/24 dev cbr0
```

## Configure IP tables

On each node run the following command to allow containers to reach the internet.

```
iptables -t nat -A POSTROUTING -o ens33 -j MASQUERADE \! -d 10.0.0.0/8
```
