# Proxy

etcd can run as a transparent proxy. Doing so allows for easy discovery of etcd within your infrastructure, since it can run on each machine as a local service. In this mode, etcd acts as a reverse proxy and forwards client requests to an active etcd cluster. The etcd proxy does not participate in the consensus replication of the etcd cluster, thus it neither increases the resilience nor decreases the write performance of the etcd cluster.

etcd currently supports two proxy modes: `readwrite` and `readonly`. The default mode is `readwrite`, which forwards both read and write requests to the etcd cluster. A `readonly` etcd proxy only forwards read requests to the etcd cluster, and returns `HTTP 501` to all write requests.

The proxy will shuffle the list of cluster members periodically to avoid sending all connections to a single member.

The member list used by an etcd proxy consists of all client URLs advertised in the cluster. These client URLs are specified in each etcd cluster member's `advertise-client-urls` option.

An etcd proxy examines several command-line options to discover its peer URLs. In order of precedence, these options are `discovery`, `discovery-srv`, and `initial-cluster`. The `initial-cluster` option is set to a comma-separated list of one or more etcd peer URLs used temporarily in order to discover the permanent cluster.

After establishing a list of peer URLs in this manner, the proxy retrieves the list of client URLs from the first reachable peer. These client URLs are specified by the `advertise-client-urls` option to etcd peers. The proxy then continues to connect to the first reachable etcd cluster member every thirty seconds to refresh the list of client URLs.

While etcd proxies therefore do not need to be given the `advertise-client-urls` option, as they retrieve this configuration from the cluster, this implies that `initial-cluster` must be set correctly for every proxy, and the `advertise-client-urls` option must be set correctly for every non-proxy, first-order cluster peer. Otherwise, requests to any etcd proxy would be forwarded improperly. Take special care not to set the `advertise-client-urls` option to URLs that point to the proxy itself, as such a configuration will cause the proxy to enter a loop, forwarding requests to itself until resources are exhausted. To correct either case, stop etcd and restart it with the correct URLs.

[This example Procfile][procfile] illustrates the difference in the etcd peer and proxy command lines used to configure and start a cluster with one proxy under the [goreman process management utility][goreman].

To summarize etcd proxy startup and peer discovery:

1. etcd proxies execute the following steps in order until the cluster *peer-urls* are known:
	1. If `discovery` is set for the proxy, ask the given discovery service for
	   the *peer-urls*. The *peer-urls* will be the combined
	   `initial-advertise-peer-urls` of all first-order, non-proxy cluster
	   members.
	2. If `discovery-srv` is set for the proxy, the *peer-urls* are discovered
	   from DNS.
	3. If `initial-cluster` is set for the proxy, that will become the value of
	   *peer-urls*.
	4. Otherwise use the default value of
	   `http://localhost:2380,http://localhost:7001`.
2. These *peer-urls* are used to contact the (non-proxy) members of the cluster
   to find their *client-urls*. The *client-urls* will thus be the combined
   `advertise-client-urls` of all cluster members (i.e. non-proxies).
3. Request of clients of the proxy will be forwarded (proxied) to these
   *client-urls*.

Always start the first-order etcd cluster members first, then any proxies. A proxy must be able to reach the cluster members to retrieve its configuration, and will attempt connections somewhat aggressively in the absence of such a channel. Starting the members before any proxy ensures the proxy can discover the client URLs when it later starts.

## Using an etcd proxy
To start etcd in proxy mode, you need to provide three flags: `proxy`, `listen-client-urls`, and `initial-cluster` (or `discovery`).

To start a readwrite proxy, set `-proxy on`; To start a readonly proxy, set `-proxy readonly`.

The proxy will be listening on `listen-client-urls` and forward requests to the etcd cluster discovered from in `initial-cluster` or `discovery` url.

### Start an etcd proxy with a static configuration
To start a proxy that will connect to a statically defined etcd cluster, specify the `initial-cluster` flag:

```
etcd --proxy on \
--listen-client-urls http://127.0.0.1:2379 \
--initial-cluster infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380,infra2=http://10.0.1.12:2380
```

### Start an etcd proxy with the discovery service
If you bootstrap an etcd cluster using the [discovery service][discovery-service], you can also start the proxy with the same `discovery`.

To start a proxy using the discovery service, specify the `discovery` flag. The proxy will wait until the etcd cluster defined at the `discovery` url finishes bootstrapping, and then start to forward the requests.

```
etcd --proxy on \
--listen-client-urls http://127.0.0.1:2379 \
--discovery https://discovery.etcd.io/3e86b59982e49066c5d813af1c2e2579cbf573de \
```

## Fallback to proxy mode with discovery service

If you bootstrap an etcd cluster using [discovery service][discovery-service] with more than the expected number of etcd members, the extra etcd processes will fall back to being `readwrite` proxies by default. They will forward the requests to the cluster as described above. For example, if you create a discovery url with `size=5`, and start ten etcd processes using that same discovery url, the result will be a cluster with five etcd members and five proxies. Note that this behaviour can be disabled with the `discovery-fallback='exit'` flag.

## Promote a proxy to a member of etcd cluster

A Proxy is in the part of etcd cluster that does not participate in consensus. A proxy will not promote itself to an etcd member that participates in consensus automatically in any case.

If you want to promote a proxy to an etcd member, there are four steps you need to follow:

- use etcdctl to add the proxy node as an etcd member into the existing cluster
- stop the etcd proxy process or service
- remove the existing proxy data directory
- restart the etcd process with new member configuration

## Example

We assume you have a one member etcd cluster with one proxy. The cluster information is listed below:

|Name|Address|
|------|---------|
|infra0|10.0.1.10|
|proxy0|10.0.1.11|

This example walks you through a case that you promote one proxy to an etcd member. The cluster will become a two member cluster after finishing the four steps.

### Add a new member into the existing cluster

First, use etcdctl to add the member to the cluster, which will output the environment variables need to correctly configure the new member:

``` bash
$ etcdctl -endpoint http://10.0.1.10:2379 member add infra1 http://10.0.1.11:2380
added member 9bf1b35fc7761a23 to cluster

ETCD_NAME="infra1"
ETCD_INITIAL_CLUSTER="infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380"
ETCD_INITIAL_CLUSTER_STATE=existing
```

### Stop the proxy process

Stop the existing proxy so we can wipe its state on disk and reload it with the new configuration:

``` bash
px aux | grep etcd
kill %etcd_proxy_pid%
```

or (if you are running etcd proxy as etcd service under systemd)

``` bash
sudo systemctl stop etcd
```

### Remove the existing proxy data dir

``` bash
rm -rf %data_dir%/proxy
```

### Start etcd as a new member

Finally, start the reconfigured member and make sure it joins the cluster correctly:

``` bash
$ export ETCD_NAME="infra1"
$ export ETCD_INITIAL_CLUSTER="infra0=http://10.0.1.10:2380,infra1=http://10.0.1.11:2380"
$ export ETCD_INITIAL_CLUSTER_STATE=existing
$ etcd --listen-client-urls http://10.0.1.11:2379 \
--advertise-client-urls http://10.0.1.11:2379 \
--listen-peer-urls http://10.0.1.11:2380 \
--initial-advertise-peer-urls http://10.0.1.11:2380 \
--data-dir %data_dir%
```

If you are running etcd under systemd, you should modify the service file with correct configuration and restart the service:

``` bash
sudo systemd restart etcd
```

If an error occurs, check the [add member troubleshooting doc][runtime-configuration].

[discovery-service]: clustering.md#discovery
[goreman]: https://github.com/mattn/goreman
[procfile]: https://github.com/coreos/etcd/blob/master/Procfile
[runtime-configuration]: runtime-configuration.md#error-cases-when-adding-members
