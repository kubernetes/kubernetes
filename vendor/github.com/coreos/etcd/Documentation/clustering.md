# Clustering Guide

## Overview

Starting an etcd cluster statically requires that each member knows another in the cluster. In a number of cases, you might not know the IPs of your cluster members ahead of time. In these cases, you can bootstrap an etcd cluster with the help of a discovery service.

Once an etcd cluster is up and running, adding or removing members is done via [runtime reconfiguration][runtime-conf]. To better understand the design behind runtime reconfiguration, we suggest you read [the runtime configuration design document][runtime-reconf-design].

This guide will cover the following mechanisms for bootstrapping an etcd cluster:

* [Static](#static)
* [etcd Discovery](#etcd-discovery)
* [DNS Discovery](#dns-discovery)

Each of the bootstrapping mechanisms will be used to create a three machine etcd cluster with the following details:

|Name|Address|Hostname|
|------|---------|------------------|
|infra0|10.0.1.10|infra0.example.com|
|infra1|10.0.1.11|infra1.example.com|
|infra2|10.0.1.12|infra2.example.com|

## Static

As we know the cluster members, their addresses and the size of the cluster before starting, we can use an offline bootstrap configuration by setting the `initial-cluster` flag. Each machine will get either the following command line or environment variables:

```
ETCD_INITIAL_CLUSTER="infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380,infra2=http://10.0.1.12:2380"
ETCD_INITIAL_CLUSTER_STATE=new
```

```
--initial-cluster infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380,infra2=http://10.0.1.12:2380 \
--initial-cluster-state new
```

Note that the URLs specified in `initial-cluster` are the _advertised peer URLs_, i.e. they should match the value of `initial-advertise-peer-urls` on the respective nodes.

If you are spinning up multiple clusters (or creating and destroying a single cluster) with same configuration for testing purpose, it is highly recommended that you specify a unique `initial-cluster-token` for the different clusters. By doing this, etcd can generate unique cluster IDs and member IDs for the clusters even if they otherwise have the exact same configuration. This can protect you from cross-cluster-interaction, which might corrupt your clusters.

etcd listens on [`listen-client-urls`][conf-listen-client] to accept client traffic. etcd member advertises the URLs specified in [`advertise-client-urls`][conf-adv-client] to other members, proxies, clients. Please make sure the `advertise-client-urls` are reachable from intended clients. A common mistake is setting `advertise-client-urls` to localhost or leave it as default when you want the remote clients to reach etcd.

On each machine you would start etcd with these flags:

```
$ etcd --name infra0 --initial-advertise-peer-urls http://10.0.1.10:2380 \
  --listen-peer-urls http://10.0.1.10:2380 \
  --listen-client-urls http://10.0.1.10:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.10:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380,infra2=http://10.0.1.12:2380 \
  --initial-cluster-state new
```
```
$ etcd --name infra1 --initial-advertise-peer-urls http://10.0.1.11:2380 \
  --listen-peer-urls http://10.0.1.11:2380 \
  --listen-client-urls http://10.0.1.11:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.11:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380,infra2=http://10.0.1.12:2380 \
  --initial-cluster-state new
```
```
$ etcd --name infra2 --initial-advertise-peer-urls http://10.0.1.12:2380 \
  --listen-peer-urls http://10.0.1.12:2380 \
  --listen-client-urls http://10.0.1.12:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.12:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380,infra2=http://10.0.1.12:2380 \
  --initial-cluster-state new
```

The command line parameters starting with `--initial-cluster` will be ignored on subsequent runs of etcd. You are free to remove the environment variables or command line flags after the initial bootstrap process. If you need to make changes to the configuration later (for example, adding or removing members to/from the cluster), see the [runtime configuration][runtime-conf] guide.

### Error Cases

In the following example, we have not included our new host in the list of enumerated nodes. If this is a new cluster, the node _must_ be added to the list of initial cluster members.

```
$ etcd --name infra1 --initial-advertise-peer-urls http://10.0.1.11:2380 \
  --listen-peer-urls https://10.0.1.11:2380 \
  --listen-client-urls http://10.0.1.11:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.11:2379 \
  --initial-cluster infra0=http://10.0.1.10:2380 \
  --initial-cluster-state new
etcd: infra1 not listed in the initial cluster config
exit 1
```

In this example, we are attempting to map a node (infra0) on a different address (127.0.0.1:2380) than its enumerated address in the cluster list (10.0.1.10:2380). If this node is to listen on multiple addresses, all addresses _must_ be reflected in the "initial-cluster" configuration directive.

```
$ etcd --name infra0 --initial-advertise-peer-urls http://127.0.0.1:2380 \
  --listen-peer-urls http://10.0.1.10:2380 \
  --listen-client-urls http://10.0.1.10:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.10:2379 \
  --initial-cluster infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380,infra2=http://10.0.1.12:2380 \
  --initial-cluster-state=new
etcd: error setting up initial cluster: infra0 has different advertised URLs in the cluster and advertised peer URLs list
exit 1
```

If you configure a peer with a different set of configuration and attempt to join this cluster you will get a cluster ID mismatch and etcd will exit.

```
$ etcd --name infra3 --initial-advertise-peer-urls http://10.0.1.13:2380 \
  --listen-peer-urls http://10.0.1.13:2380 \
  --listen-client-urls http://10.0.1.13:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.13:2379 \
  --initial-cluster infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380,infra3=http://10.0.1.13:2380 \
  --initial-cluster-state=new
etcd: conflicting cluster ID to the target cluster (c6ab534d07e8fcc4 != bc25ea2a74fb18b0). Exiting.
exit 1
```

## Discovery

In a number of cases, you might not know the IPs of your cluster peers ahead of time. This is common when utilizing cloud providers or when your network uses DHCP. In these cases, rather than specifying a static configuration, you can use an existing etcd cluster to bootstrap a new one. We call this process "discovery".

There two methods that can be used for discovery:

* etcd discovery service
* DNS SRV records

### etcd Discovery

To better understand the design about discovery service protocol, we suggest you read [this][discovery-proto].

#### Lifetime of a Discovery URL

A discovery URL identifies a unique etcd cluster. Instead of reusing a discovery URL, you should always create discovery URLs for new clusters.

Moreover, discovery URLs should ONLY be used for the initial bootstrapping of a cluster. To change cluster membership after the cluster is already running, see the [runtime reconfiguration][runtime-conf] guide.

#### Custom etcd Discovery Service

Discovery uses an existing cluster to bootstrap itself. If you are using your own etcd cluster you can create a URL like so:

```
$ curl -X PUT https://myetcd.local/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0fb83/_config/size -d value=3
```

By setting the size key to the URL, you create a discovery URL with an expected cluster size of 3.

If you bootstrap an etcd cluster using discovery service with more than the expected number of etcd members, the extra etcd processes will [fall back][fall-back] to being [proxies][proxy] by default.

The URL you will use in this case will be `https://myetcd.local/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0fb83` and the etcd members will use the `https://myetcd.local/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0fb83` directory for registration as they start.

**Each member must have a different name flag specified. `Hostname` or `machine-id` can be a good choice. Or discovery will fail due to duplicated name.**

Now we start etcd with those relevant flags for each member:

```
$ etcd --name infra0 --initial-advertise-peer-urls http://10.0.1.10:2380 \
  --listen-peer-urls http://10.0.1.10:2380 \
  --listen-client-urls http://10.0.1.10:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.10:2379 \
  --discovery https://myetcd.local/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0fb83
```
```
$ etcd --name infra1 --initial-advertise-peer-urls http://10.0.1.11:2380 \
  --listen-peer-urls http://10.0.1.11:2380 \
  --listen-client-urls http://10.0.1.11:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.11:2379 \
  --discovery https://myetcd.local/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0fb83
```
```
$ etcd --name infra2 --initial-advertise-peer-urls http://10.0.1.12:2380 \
  --listen-peer-urls http://10.0.1.12:2380 \
  --listen-client-urls http://10.0.1.12:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.12:2379 \
  --discovery https://myetcd.local/v2/keys/discovery/6c007a14875d53d9bf0ef5a6fc0257c817f0fb83
```

This will cause each member to register itself with the custom etcd discovery service and begin the cluster once all machines have been registered.

#### Public etcd Discovery Service

If you do not have access to an existing cluster, you can use the public discovery service hosted at `discovery.etcd.io`.  You can create a private discovery URL using the "new" endpoint like so:

```
$ curl https://discovery.etcd.io/new?size=3
https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de
```

This will create the cluster with an initial expected size of 3 members. If you do not specify a size, a default of 3 will be used.

If you bootstrap an etcd cluster using discovery service with more than the expected number of etcd members, the extra etcd processes will [fall back][fall-back] to being [proxies][proxy] by default.

```
ETCD_DISCOVERY=https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de
```

```
-discovery https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de
```

**Each member must have a different name flag specified. `Hostname` or `machine-id` can be a good choice. Or discovery will fail due to duplicated name.**

Now we start etcd with those relevant flags for each member:

```
$ etcd --name infra0 --initial-advertise-peer-urls http://10.0.1.10:2380 \
  --listen-peer-urls http://10.0.1.10:2380 \
  --listen-client-urls http://10.0.1.10:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.10:2379 \
  --discovery https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de
```
```
$ etcd --name infra1 --initial-advertise-peer-urls http://10.0.1.11:2380 \
  --listen-peer-urls http://10.0.1.11:2380 \
  --listen-client-urls http://10.0.1.11:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.11:2379 \
  --discovery https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de
```
```
$ etcd --name infra2 --initial-advertise-peer-urls http://10.0.1.12:2380 \
  --listen-peer-urls http://10.0.1.12:2380 \
  --listen-client-urls http://10.0.1.12:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.12:2379 \
  --discovery https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de
```

This will cause each member to register itself with the discovery service and begin the cluster once all members have been registered.

You can use the environment variable `ETCD_DISCOVERY_PROXY` to cause etcd to use an HTTP proxy to connect to the discovery service.

#### Error and Warning Cases

##### Discovery Server Errors


```
$ etcd --name infra0 --initial-advertise-peer-urls http://10.0.1.10:2380 \
  --listen-peer-urls http://10.0.1.10:2380 \
  --listen-client-urls http://10.0.1.10:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.10:2379 \
  --discovery https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de
etcd: error: the cluster doesn’t have a size configuration value in https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de/_config
exit 1
```

##### User Errors

This error will occur if the discovery cluster already has the configured number of members, and `discovery-fallback` is explicitly disabled

```
$ etcd --name infra0 --initial-advertise-peer-urls http://10.0.1.10:2380 \
  --listen-peer-urls http://10.0.1.10:2380 \
  --listen-client-urls http://10.0.1.10:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.10:2379 \
  --discovery https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de \
  --discovery-fallback exit
etcd: discovery: cluster is full
exit 1
```

##### Warnings

This is a harmless warning notifying you that the discovery URL will be
ignored on this machine.

```
$ etcd --name infra0 --initial-advertise-peer-urls http://10.0.1.10:2380 \
  --listen-peer-urls http://10.0.1.10:2380 \
  --listen-client-urls http://10.0.1.10:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.0.1.10:2379 \
  --discovery https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de
etcdserver: discovery token ignored since a cluster has already been initialized. Valid log found at /var/lib/etcd
```

### DNS Discovery

DNS [SRV records][rfc-srv] can be used as a discovery mechanism.
The `-discovery-srv` flag can be used to set the DNS domain name where the discovery SRV records can be found.
The following DNS SRV records are looked up in the listed order:

* _etcd-server-ssl._tcp.example.com
* _etcd-server._tcp.example.com

If `_etcd-server-ssl._tcp.example.com` is found then etcd will attempt the bootstrapping process over SSL.

To help clients discover the etcd cluster, the following DNS SRV records are looked up in the listed order:

* _etcd-client._tcp.example.com
* _etcd-client-ssl._tcp.example.com

If `_etcd-client-ssl._tcp.example.com` is found, clients will attempt to communicate with the etcd cluster over SSL.

#### Create DNS SRV records

```
$ dig +noall +answer SRV _etcd-server._tcp.example.com
_etcd-server._tcp.example.com. 300 IN  SRV  0 0 2380 infra0.example.com.
_etcd-server._tcp.example.com. 300 IN  SRV  0 0 2380 infra1.example.com.
_etcd-server._tcp.example.com. 300 IN  SRV  0 0 2380 infra2.example.com.
```

```
$ dig +noall +answer SRV _etcd-client._tcp.example.com
_etcd-client._tcp.example.com. 300 IN SRV 0 0 2379 infra0.example.com.
_etcd-client._tcp.example.com. 300 IN SRV 0 0 2379 infra1.example.com.
_etcd-client._tcp.example.com. 300 IN SRV 0 0 2379 infra2.example.com.
```

```
$ dig +noall +answer infra0.example.com infra1.example.com infra2.example.com
infra0.example.com.  300  IN  A  10.0.1.10
infra1.example.com.  300  IN  A  10.0.1.11
infra2.example.com.  300  IN  A  10.0.1.12
```
#### Bootstrap the etcd cluster using DNS

etcd cluster members can listen on domain names or IP address, the bootstrap process will resolve DNS A records.

The resolved address in `--initial-advertise-peer-urls` *must match* one of the resolved addresses in the SRV targets. The etcd member reads the resolved address to find out if it belongs to the cluster defined in the SRV records.

```
$ etcd --name infra0 \
--discovery-srv example.com \
--initial-advertise-peer-urls http://infra0.example.com:2380 \
--initial-cluster-token etcd-cluster-1 \
--initial-cluster-state new \
--advertise-client-urls http://infra0.example.com:2379 \
--listen-client-urls http://infra0.example.com:2379 \
--listen-peer-urls http://infra0.example.com:2380
```

```
$ etcd --name infra1 \
--discovery-srv example.com \
--initial-advertise-peer-urls http://infra1.example.com:2380 \
--initial-cluster-token etcd-cluster-1 \
--initial-cluster-state new \
--advertise-client-urls http://infra1.example.com:2379 \
--listen-client-urls http://infra1.example.com:2379 \
--listen-peer-urls http://infra1.example.com:2380
```

```
$ etcd --name infra2 \
--discovery-srv example.com \
--initial-advertise-peer-urls http://infra2.example.com:2380 \
--initial-cluster-token etcd-cluster-1 \
--initial-cluster-state new \
--advertise-client-urls http://infra2.example.com:2379 \
--listen-client-urls http://infra2.example.com:2379 \
--listen-peer-urls http://infra2.example.com:2380
```

You can also bootstrap the cluster using IP addresses instead of domain names:

```
$ etcd --name infra0 \
--discovery-srv example.com \
--initial-advertise-peer-urls http://10.0.1.10:2380 \
--initial-cluster-token etcd-cluster-1 \
--initial-cluster-state new \
--advertise-client-urls http://10.0.1.10:2379 \
--listen-client-urls http://10.0.1.10:2379 \
--listen-peer-urls http://10.0.1.10:2380
```

```
$ etcd --name infra1 \
--discovery-srv example.com \
--initial-advertise-peer-urls http://10.0.1.11:2380 \
--initial-cluster-token etcd-cluster-1 \
--initial-cluster-state new \
--advertise-client-urls http://10.0.1.11:2379 \
--listen-client-urls http://10.0.1.11:2379 \
--listen-peer-urls http://10.0.1.11:2380
```

```
$ etcd --name infra2 \
--discovery-srv example.com \
--initial-advertise-peer-urls http://10.0.1.12:2380 \
--initial-cluster-token etcd-cluster-1 \
--initial-cluster-state new \
--advertise-client-urls http://10.0.1.12:2379 \
--listen-client-urls http://10.0.1.12:2379 \
--listen-peer-urls http://10.0.1.12:2380
```

#### etcd proxy configuration

DNS SRV records can also be used to configure the list of peers for an etcd server running in proxy mode:

```
$ etcd --proxy on --discovery-srv example.com
```

#### etcd client configuration

DNS SRV records can also be used to help clients discover the etcd cluster.

The official [etcd/client][client] supports [DNS Discovery][client-discoverer].

`etcdctl` also supports DNS Discovery by specifying the `--discovery-srv` option.

```
$ etcdctl --discovery-srv example.com set foo bar
```

#### Error Cases

You might see an error like `cannot find local etcd $name from SRV records.`. That means the etcd member fails to find itself from the cluster defined in SRV records. The resolved address in `--initial-advertise-peer-urls` *must match* one of the resolved addresses in the SRV targets.

# 0.4 to 2.0+ Migration Guide

In etcd 2.0 we introduced the ability to listen on more than one address and to advertise multiple addresses. This makes using etcd easier when you have complex networking, such as private and public networks on various cloud providers.

To make understanding this feature easier, we changed the naming of some flags, but we support the old flags to make the migration from the old to new version easier.

|Old Flag    |New Flag    |Migration Behavior                  |
|-----------------------|-----------------------|---------------------------------------------------------------------------------------|
|-peer-addr    |--initial-advertise-peer-urls   |If specified, peer-addr will be used as the only peer URL. Error if both flags specified.|
|-addr      |--advertise-client-urls  |If specified, addr will be used as the only client URL. Error if both flags specified.|
|-peer-bind-addr  |--listen-peer-urls  |If specified, peer-bind-addr will be used as the only peer bind URL. Error if both flags specified.|
|-bind-addr    |--listen-client-urls  |If specified, bind-addr will be used as the only client bind URL. Error if both flags specified.|
|-peers      |none      |Deprecated. The --initial-cluster flag provides a similar concept with different semantics. Please read this guide on cluster startup.|
|-peers-file    |none      |Deprecated. The --initial-cluster flag provides a similar concept with different semantics. Please read this guide on cluster startup.|

[client]: /client
[client-discoverer]: https://godoc.org/github.com/coreos/etcd/client#Discoverer
[conf-adv-client]: configuration.md#-advertise-client-urls
[conf-listen-client]: configuration.md#-listen-client-urls
[discovery-proto]: discovery_protocol.md
[fall-back]: proxy.md#fallback-to-proxy-mode-with-discovery-service
[proxy]: proxy.md
[rfc-srv]: http://www.ietf.org/rfc/rfc2052.txt
[runtime-conf]: runtime-configuration.md
[runtime-reconf-design]: runtime-reconf-design.md
