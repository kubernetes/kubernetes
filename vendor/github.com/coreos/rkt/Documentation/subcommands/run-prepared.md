# rkt run-prepared

Once a pod is prepared with rkt [prepare][prepare], it can be run by executing `rkt run-prepared UUID`.

## Example

```
# rkt list
UUID        APP ACI     STATE       NETWORKS
c9fad0e6    etcd    coreos.com/etcd prepared
# rkt run-prepared c9fad0e6
2015/10/01 16:44:08 Setting up stage1
2015/10/01 16:44:08 Wrote filesystem to /var/lib/rkt/pods/run/c9fad0e6-8236-4fc2-ad17-55d0a4c7d742
2015/10/01 16:44:08 Pivoting to filesystem /var/lib/rkt/pods/run/c9fad0e6-8236-4fc2-ad17-55d0a4c7d742
2015/10/01 16:44:08 Execing /init
[25701.705171] etcd[4]: 2015/10/01 14:44:09 etcd: no data-dir provided, using default data-dir ./default.etcd
[25701.705596] etcd[4]: 2015/10/01 14:44:09 etcd: listening for peers on http://localhost:2380
[25701.705875] etcd[4]: 2015/10/01 14:44:09 etcd: listening for peers on http://localhost:7001
[25701.706473] etcd[4]: 2015/10/01 14:44:09 etcd: listening for client requests on http://localhost:2379
[25701.706679] etcd[4]: 2015/10/01 14:44:09 etcd: listening for client requests on http://localhost:4001
[25701.706842] etcd[4]: 2015/10/01 14:44:09 etcdserver: datadir is valid for the 2.0.1 format
[25701.706999] etcd[4]: 2015/10/01 14:44:09 etcdserver: name = default
[25701.707147] etcd[4]: 2015/10/01 14:44:09 etcdserver: data dir = default.etcd
[25701.707294] etcd[4]: 2015/10/01 14:44:09 etcdserver: member dir = default.etcd/member
[25701.707464] etcd[4]: 2015/10/01 14:44:09 etcdserver: heartbeat = 100ms
[25701.707624] etcd[4]: 2015/10/01 14:44:09 etcdserver: election = 1000ms
[25701.707771] etcd[4]: 2015/10/01 14:44:09 etcdserver: snapshot count = 10000
[25701.707917] etcd[4]: 2015/10/01 14:44:09 etcdserver: advertise client URLs = http://localhost:2379,http://localhost:4001
[25701.708062] etcd[4]: 2015/10/01 14:44:09 etcdserver: initial advertise peer URLs = http://localhost:2380,http://localhost:7001
[25701.708216] etcd[4]: 2015/10/01 14:44:09 etcdserver: initial cluster = default=http://localhost:2380,default=http://localhost:7001
[25701.712024] etcd[4]: 2015/10/01 14:44:09 etcdserver: start member ce2a822cea30bfca in cluster 7e27652122e8b2ae
[25701.712623] etcd[4]: 2015/10/01 14:44:09 raft: ce2a822cea30bfca became follower at term 0
[25701.713183] etcd[4]: 2015/10/01 14:44:09 raft: newRaft ce2a822cea30bfca [peers: [], term: 0, commit: 0, applied: 0, lastindex: 0, lastterm: 0]
[25701.713378] etcd[4]: 2015/10/01 14:44:09 raft: ce2a822cea30bfca became follower at term 1
[25701.716177] etcd[4]: 2015/10/01 14:44:09 etcdserver: added local member ce2a822cea30bfca [http://localhost:2380 http://localhost:7001] to cluster 7e27652122e8b2ae
[25703.012367] etcd[4]: 2015/10/01 14:44:10 raft: ce2a822cea30bfca is starting a new election at term 1
[25703.012749] etcd[4]: 2015/10/01 14:44:10 raft: ce2a822cea30bfca became candidate at term 2
[25703.012976] etcd[4]: 2015/10/01 14:44:10 raft: ce2a822cea30bfca received vote from ce2a822cea30bfca at term 2
[25703.013193] etcd[4]: 2015/10/01 14:44:10 raft: ce2a822cea30bfca became leader at term 2
[25703.013405] etcd[4]: 2015/10/01 14:44:10 raft.node: ce2a822cea30bfca elected leader ce2a822cea30bfca at term 2
[25703.017089] etcd[4]: 2015/10/01 14:44:10 etcdserver: published {Name:default ClientURLs:[http://localhost:2379 http://localhost:4001]} to cluster 7e27652122e8b2ae
```

## Options

| Flag | Default | Options | Description |
| --- | --- | --- | --- |
| `--dns` |  `` | IP Address | Name server to write in `/etc/resolv.conf`. It can be specified several times |
| `--dns-opt` |  `` | Option as described in the options section in resolv.conf(5) | DNS option to write in `/etc/resolv.conf`. It can be specified several times |
| `--dns-search` |  `` | Domain name | DNS search domain to write in `/etc/resolv.conf`. It can be specified several times |
| `--hostname` | "rkt-$PODUUID" | A host name | Set pod's host name. |
| `--interactive` |  `false` | `true` or `false` | Run pod interactively. If true, only one image may be supplied |
| `--mds-register` |  `false` | `true` or `false` | Register pod with metadata service. It needs network connectivity to the host (`--net=(default|default-restricted|host)` |
| `--net` |  `default` | A comma-separated list of networks. Syntax: `--net[=n[:args], ...]` | Configure the pod's networking. Optionally, pass a list of user-configured networks to load and set arguments to pass to each network, respectively |

## Global options

See the table with [global options in general commands documentation][global-options].


[global-options]: ../commands.md#global-options
[prepare]: prepare.md
