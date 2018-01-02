# Production users

This document tracks people and use cases for etcd in production. By creating a list of production use cases we hope to build a community of advisors that we can reach out to with experience using various etcd applications, operation environments, and cluster sizes. The etcd development team may reach out periodically to check-in on how etcd is working in the field and update this list.

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

## Vonage

- *Application*: system configuration for microservices, scheduling, locks (future - service discovery)
- *Launched*: August 2015
- *Cluster Size*: 2 clusters of 5 members in 2 DCs, n local proxies 1-to-1 with microservice, (ssl and SRV look up)
- *Order of Data Size*: kilobytes
- *Operator*: Vonage [devAdmin][raoofm]
- *Environment*: VMWare, AWS
- *Backups*: Daily snapshots on VMs. Backups done for upgrades.

[teamcity]: https://www.jetbrains.com/teamcity/
[raoofm]:https://github.com/raoofm

## Qiniu Cloud

- *Application*: system configuration for microservices, distributed locks
- *Launched*: Jan. 2016
- *Cluster Size*: 3 members each with several clusters
- *Order of Data Size*: kilobytes
- *Operator*: Pandora, chenchao@qiniu.com
- *Environment*: Baremetal
- *Backups*: None, all data can be recreated if necessary

## QingCloud

- *Application*: [QingCloud][qingcloud] appcenter cluster for service discovery as [metad][metad] backend.
- *Launched*: December 2016
- *Cluster Size*: 1 cluster of 3 members per user.
- *Order of Data Size*: kilobytes
- *Operator*: [yunify][yunify]
- *Environment*: QingCloud IaaS
- *Backups*: None, all data can be recreated if necessary.

[metad]:https://github.com/yunify/metad
[yunify]:https://github.com/yunify
[qingcloud]:https://qingcloud.com/


## Yandex

- *Application*: system configuration for services, service discovery
- *Launched*: March 2016
- *Cluster Size*: 3 clusters of 5 members
- *Order of Data Size*: several gigabytes
- *Operator*: Yandex; [nekto0n][nekto0n]
- *Environment*: Bare Metal
- *Backups*: None

[nekto0n]:https://github.com/nekto0n

## Tencent Games

- *Application*: Meta data and configuration data for service discovery, Kubernetes, etc.
- *Launched*: Jan. 2015
- *Cluster Size*: 3 members each with 10s of clusters
- *Order of Data Size*: 10s of Megabytes
- *Operator*: Tencent Game Operations Department
- *Environment*: Baremetal
- *Backups*: Periodic sync to backup server

In Tencent games, we use Docker and Kubernetes to deploy and run our applications, and use etcd to save meta data for service discovery, Kubernetes, etc.

## Hyper.sh

- *Application*: Kubernetes, distributed locks, etc.
- *Launched*: April 2016
- *Cluster Size*: 1 cluster of 3 members
- *Order of Data Size*: 10s of MB
- *Operator*: Hyper.sh
- *Environment*: Baremetal
- *Backups*: None, all data can be recreated if necessary.

In [hyper.sh][hyper.sh], the container service is backed by [hypernetes][hypernetes], a multi-tenant kubernetes distro. Moreover, we use etcd to coordinate the multiple manage services and store global meta data.

[hypernetes]:https://github.com/hyperhq/hypernetes
[Hyper.sh]:https://www.hyper.sh

## Meitu
- *Application*: system configuration for services, service discovery, kubernetes in test environment
- *Launched*: October 2015
- *Cluster Size*: 1 cluster of 3 members
- *Order of Data Size*: megabytes
- *Operator*: Meitu, hxj@meitu.com, [shafreeck][shafreeck]
- *Environment*: Bare Metal
- *Backups*: None, all data can be recreated if necessary.

[shafreeck]:https://github.com/shafreeck

## Grab
- *Application*: system configuration for services, service discovery
- *Launched*: June 2016
- *Cluster Size*: 1 cluster of 7 members
- *Order of Data Size*: megabytes
- *Operator*: Grab, [taxitan][taxitan], [reterVision][reterVision]
- *Environment*: AWS
- *Backups*: None, all data can be recreated if necessary.

[taxitan]:https://github.com/taxitan
[reterVision]:https://github.com/reterVision

## DaoCloud.io

- *Application*: container management
- *Launched*: Sep. 2015
- *Cluster Size*: 1000+ deployments, each deployment contains a 3 node cluster.
- *Order of Data Size*: 100s of Megabytes
- *Operator*: daocloud.io
- *Environment*: Baremetal and virtual machines
- *Backups*: None, all data can be recreated if necessary.

In [DaoCloud][DaoCloud], we use Docker and Swarm to deploy and run our applications, and we use etcd to save metadata for service discovery.

[DaoCloud]:https://www.daocloud.io

## Branch.io

- *Application*: Kubernetes
- *Launched*: April 2016
- *Cluster Size*: Multiple clusters, multiple sizes
- *Order of Data Size*: 100s of Megabytes
- *Operator*: branch.io
- *Environment*: AWS, Kubernetes
- *Backups*: EBS volume backups

At [Branch][branch], we use kubernetes heavily as our core microservice platform for staging and production.

[branch]: https://branch.io

## Baidu Waimai

- *Application*: SkyDNS, Kubernetes, UDC, CMDB and other distributed systems
- *Launched*: April. 2016
- *Cluster Size*: 3 clusters of 5 members
- *Order of Data Size*: several gigabytes
- *Operator*: Baidu Waimai Operations Department
- *Environment*: CentOS 6.5
- *Backups*: backup scripts

## Salesforce.com

- *Application*: Kubernetes
- *Launched*: Jan 2017
- *Cluster Size*: Multiple clusters of 3 members
- *Order of Data Size*: 100s of Megabytes
- *Operator*: Salesforce.com (krmayankk@github)
- *Environment*: BareMetal
- *Backups*: None, all data can be recreated

## Hosted Graphite

- *Application*: Service discovery, locking, ephemeral application data
- *Launched*: January 2017
- *Cluster Size*: 2 clusters of 7 members
- *Order of Data Size*: Megabytes
- *Operator*: Hosted Graphite (sre@hostedgraphite.com)
- *Environment*: Bare Metal
- *Backups*: None, all data is considered ephemeral.
