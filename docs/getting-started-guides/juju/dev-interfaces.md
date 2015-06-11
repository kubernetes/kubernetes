# Interfaces

Juju charms share data on relations.  Relations are a bi-directional lines
of communication between charms to exchange information.  This is the magic
part of orchestration!

## kubernetes-master charm

### Interfaces Provided
The kubernetes-master charm provides the following relations in Juju.

#### Relationship:  client-api
##### Interface:  kubernetes-client

- (str) minion_url
- (str) port

> Not currently implemented

#### Relationship:  minions-api
##### Interface:  kubernetes-api
This interface sets the private IP address of the kubernetes-master and port
8080.

- private-address
  - implicitly sent on all relations, private address of Kubernetes Master unit
- port
  - port of Kubernetes Master API, currently hard coded to 8080
- version
  - current version if deployed from a tag


### Requires Interfaces

#### Relationship: etcd
##### Interface: etcd

- private-address
  - implicitly sent on all relations, private address of ETCD server
- port
  - port of ETCD daemon, currently hard coded to 4001
- hostname
  - repeat of private-address
- public_hostname
  - public interface ip address of ETCD server


## kubernetes charm

### Interfaces Required
The kubernetes charm requires the following relations in Juju.

#### Relationship:  etcd
##### Interface: etcd


- private-address
  - implicitly sent on all relations, private address of ETCD server
- port
  - port of ETCD daemon, currently hard coded to 4001
- hostname
  - repeat of private-address
- public_hostname
  - public interface ip address of ETCD server

#### Relationship: api
##### Interface: kubernetes-api

This interface sets the private IP address of the kubernetes-master and port
8080.

- private-address
  - implicitly sent on all relations, private address of Kubernetes Master unit
- port
  - port of Kubernetes Master API, currently hard coded to 8080
- version
  - current version if deployed from a tag



#### Relationship: network
##### Interface: overlay-network

- flanel_mtu
  - MTU for the virtual network interface
- flannel_subnet
  - private addressing space for the virtual network interface
