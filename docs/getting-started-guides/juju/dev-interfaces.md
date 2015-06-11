# Interfaces

Juju charms share data on relations.  Relations are a bi-directional lines
of communication between charms to exchange information.  This is the magic
part of orchestration!

## kubernetes-master charm

### Interfaces Provided
The kubernetes-master charm provides the following relations in Juju.

#### Relationship:  client-api
#### Interface:  kubernetes-client

#### Relationship:  minions-api
#### Interface:  kubernetes-api
This interface sets the private IP address of the kuberentes-master and port 
8080.

### Requires Interfaces

#### Relationship: etcd
#### Interface: etcd



## kubernetes charm

### Interfaces Required
The kubernetes charm requires the following relations in Juju.

#### Relationship:  etcd
#### Interface: etcd

#### Relationship: api
#### Interface: kuberentes-api

#### Relationship: network
#### Interface: overlay-network
