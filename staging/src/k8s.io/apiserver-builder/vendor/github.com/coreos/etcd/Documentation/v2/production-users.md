# Production Users

This document tracks people and use cases for etcd in production. By creating a list of production use cases we hope to build a community of advisors that we can reach out to with experience using various etcd applications, operation environments, and cluster sizes. The etcd development team may reach out periodically to check-in on your experience and update this list.

## discovery.etcd.io

- *Application*: https://github.com/coreos/discovery.etcd.io
- *Launched*: Feb. 2014
- *Cluster Size*: 5 members, 5 discovery proxies
- *Order of Data Size*: 100s of Megabytes
- *Operator*: CoreOS, brandon.philips@coreos.com
- *Environment*: AWS
- *Backups*: Periodic async to S3

discovery.etcd.io is the longest continuously running etcd backed service that we know about. It is the basis of automatic cluster bootstrap and was launched in Feb. 2014: https://coreos.com/blog/etcd-0.3.0-released/.

## OpenTable

- *Application*: OpenTable internal service discovery and cluster configuration management
- *Launched*: May 2014
- *Cluster Size*: 3 members each in 6 independent clusters; approximately 50 nodes reading / writing
- *Order of Data Size*: 10s of MB
- *Operator*: OpenTable, Inc; sschlansker@opentable.com
- *Environment*: AWS, VMWare
- *Backups*: None, all data can be re-created if necessary.

## cycoresys.com

- *Application*: multiple
- *Launched*: Jul. 2014
- *Cluster Size*: 3 members, _n_ proxies
- *Order of Data Size*: 100s of kilobytes
- *Operator*: CyCore Systems, Inc, sys@cycoresys.com
- *Environment*: Baremetal
- *Backups*: Periodic sync to Ceph RadosGW and DigitalOcean VM

CyCore Systems provides architecture and engineering for computing systems.  This cluster provides microservices, virtual machines, databases, storage clusters to a number of clients.  It is built on CoreOS machines, with each machine in the cluster running etcd as a peer or proxy.

## Radius Intelligence

- *Application*: multiple internal tools, Kubernetes clusters, bootstrappable system configs
- *Launched*: June 2015
- *Cluster Size*: 2 clusters of 5 and 3 members; approximately a dozen nodes read/write
- *Order of Data Size*: 100s of kilobytes
- *Operator*: Radius Intelligence; jcderr@radius.com
- *Environment*: AWS, CoreOS, Kubernetes
- *Backups*: None, all data can be recreated if necessary.

Radius Intelligence uses Kubernetes running CoreOS to containerize and scale internal toolsets. Examples include running [JetBrains TeamCity][teamcity] and internal AWS security and cost reporting tools. etcd clusters back these clusters as well as provide some basic environment bootstrapping configuration keys.

[teamcity]: https://www.jetbrains.com/teamcity/
