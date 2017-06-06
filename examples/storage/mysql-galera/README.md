## Galera Replication for MySQL on Kubernetes

This document explains a simple demonstration example of running MySQL synchronous replication using Galera, specifically, Percona XtraDB cluster. The example is simplistic and used a fixed number (3) of nodes but the idea can be built upon and made more dynamic as Kubernetes matures.

### Prerequisites

This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](https://kubernetes.io/docs/getting-started-guides/) for installation instructions for your platform.

Also, this example requires the image found in the ```image``` directory. For your convenience, it is built and available on Docker's public image repository as ```capttofu/percona_xtradb_cluster_5_6```. It can also be built which would merely require that the image in the pod or replication controller files is updated.

This example was tested on OS X with a Galera cluster running on VMWare using the fine repo developed by Paulo Pires [https://github.com/pires/kubernetes-vagrant-coreos-cluster] and client programs built for OS X.

### Basic concept

The basic idea is this: three replication controllers with a single pod, corresponding services, and a single overall service to connect to all three nodes. One of the important design goals of MySQL replication and/or clustering is that you don't want a single-point-of-failure, hence the need to distribute each node or slave across hosts or even geographical locations. Kubernetes is well-suited for facilitating this design pattern using the service and replication controller configuration files in this example.

By defaults, there are only three pods (hence replication controllers) for this cluster. This number can be increased using the variable NUM_NODES, specified in the replication controller configuration file. It's important to know the number of nodes must always be odd.

When the replication controller is created, it results in the corresponding container to start, run an entrypoint script that installs the MySQL system tables, set up users, and build up a list of servers that is used with the galera parameter ```wsrep_cluster_address```.  This is a list of running nodes that galera uses for election of a node to obtain SST (Single State Transfer) from.

Note: Kubernetes best-practices is to pre-create the services for each controller, and the configuration files which contain the service and replication controller for each node, when created, will result in both a service and replication contrller running for the given node. An important thing to know is that it's important that initially pxc-node1.yaml be processed first and no other pxc-nodeN services that don't have corresponding replication controllers should exist. The reason for this is that if there is a node in ```wsrep_clsuter_address``` without a backing galera node there will be nothing to obtain SST from which will cause the node to shut itself down and the container in question to exit (and another soon relaunched, repeatedly).

First, create the overall cluster service that will be used to connect to the cluster:

```kubectl create -f examples/storage/mysql-galera/pxc-cluster-service.yaml```

Create the service and replication controller for the first node:

```kubectl create -f examples/storage/mysql-galera/pxc-node1.yaml```

### Create services and controllers for the remaining nodes

Repeat the same previous steps for ```pxc-node2``` and ```pxc-node3```

When complete, you should be able connect with a MySQL client to the IP address
 service ```pxc-cluster``` to find a working cluster

### An example of creating a cluster

Shown below are examples of Using ```kubectl``` from within the ```./examples/storage/mysql-galera``` directory, the status of the lauched replication controllers and services can be confirmed

```
$ kubectl create -f examples/storage/mysql-galera/pxc-cluster-service.yaml 
services/pxc-cluster

$ kubectl create -f examples/storage/mysql-galera/pxc-node1.yaml 
services/pxc-node1
replicationcontrollers/pxc-node1

$ kubectl create -f examples/storage/mysql-galera/pxc-node2.yaml 
services/pxc-node2
replicationcontrollers/pxc-node2

$ kubectl create -f examples/storage/mysql-galera/pxc-node3.yaml 
services/pxc-node3
replicationcontrollers/pxc-node3

```

### Confirm a running cluster

Verify everything is running:

```
$ kubectl get rc,pods,services
CONTROLLER   CONTAINER(S)   IMAGE(S)                                    SELECTOR           REPLICAS
pxc-node1    pxc-node1      capttofu/percona_xtradb_cluster_5_6:beta    name=pxc-node1     1
pxc-node2    pxc-node2      capttofu/percona_xtradb_cluster_5_6:beta    name=pxc-node2     1
pxc-node3    pxc-node3      capttofu/percona_xtradb_cluster_5_6:beta    name=pxc-node3     1
NAME              READY     STATUS    RESTARTS   AGE
pxc-node1-h6fqr   1/1       Running   0          41m
pxc-node2-sfqm6   1/1       Running   0          41m
pxc-node3-017b3   1/1       Running   0          40m
NAME          LABELS    SELECTOR           IP(S)            PORT(S)
pxc-cluster   <none>    unit=pxc-cluster   10.100.179.58    3306/TCP
pxc-node1     <none>    name=pxc-node1     10.100.217.202   3306/TCP
                                                            4444/TCP
                                                            4567/TCP
                                                            4568/TCP
pxc-node2     <none>    name=pxc-node2     10.100.47.212    3306/TCP
                                                            4444/TCP
                                                            4567/TCP
                                                            4568/TCP
pxc-node3     <none>    name=pxc-node3     10.100.200.14    3306/TCP
                                                            4444/TCP
                                                            4567/TCP
                                                            4568/TCP

```

The cluster should be ready for use!

### Connecting to the cluster

Using the name of ```pxc-cluster``` service running interactively using ```kubernetes exec```, it is possible to connect to any of the pods using the mysql client on the pod's container to verify the cluster size, which should be ```3```. In this example below, pxc-node3 replication controller is chosen, and to find out the pod name, ```kubectl get pods``` and ```awk``` are employed:

```
$ kubectl get pods|grep pxc-node3|awk '{ print $1 }'
pxc-node3-0b5mc

$ kubectl exec pxc-node3-0b5mc -i -t -- mysql -u root -p -h pxc-cluster

Enter password: 
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 5
Server version: 5.6.24-72.2-56-log Percona XtraDB Cluster (GPL), Release rel72.2, Revision 43abf03, WSREP version 25.11, wsrep_25.11

Copyright (c) 2009-2015 Percona LLC and/or its affiliates
Copyright (c) 2000, 2015, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> show status like 'wsrep_cluster_size';
+--------------------+-------+
| Variable_name      | Value |
+--------------------+-------+
| wsrep_cluster_size | 3     |
+--------------------+-------+
1 row in set (0.06 sec)

```

At this point, there is a working cluster that can begin being used via the pxc-cluster service IP address!

### TODO

This setup certainly can become more fluid and dynamic. One idea is to perhaps use an etcd container to store information about node state. Originally, there was a read-only kubernetes API available to each container but that has since been removed. Also, Kelsey Hightower is working on moving the functionality of confd to Kubernetes. This could replace the shell duct tape that builds the cluster configuration file for the image.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/storage/mysql-galera/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
