# Getting Started with rkt on Ubuntu Vivid

The following guide will show you how to build and run the sample [etcd ACI](https://github.com/coreos/etcd/releases/download/v2.0.9/etcd-v2.0.9-linux-amd64.aci) on the standard vagrantcloud.com [box for Ubuntu Vivid](https://vagrantcloud.com/ubuntu/boxes/vivid64).

## Download and start an Ubuntu Vivid box

```
vagrant init ubuntu/vivid64
vagrant up --provider virtualbox
```

## SSH into the VM and install rkt

```
vagrant ssh
sudo -s

wget https://github.com/coreos/rkt/releases/download/v1.2.1/rkt-v1.2.1.tar.gz
tar xzvf rkt-v1.2.1.tar.gz
cd rkt-v1.2.1
./rkt help
```

## Trust the CoreOS signing key

This shows how to trust the CoreOS signing key using the [`rkt trust` command](https://github.com/coreos/rkt/blob/master/Documentation/commands.md#rkt-trust).

```
./rkt trust --prefix=coreos.com/etcd
Prefix: "coreos.com/etcd"
Key: "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg"
GPG key fingerprint is: 8B86 DE38 890D DB72 9186  7B02 5210 BD88 8818 2190
	CoreOS ACI Builder <release@coreos.com>
	Are you sure you want to trust this key (yes/no)?
	yes
	Trusting "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg" for prefix "coreos.com/etcd".
	Added key for prefix "coreos.com/etcd" at "/etc/rkt/trustedkeys/prefix.d/coreos.com/etcd/8b86de38890ddb7291867b025210bd8888182190"
```

For more details on how signature verification works in rkt, see the [Signing and Verification Guide](https://github.com/coreos/rkt/blob/master/Documentation/signing-and-verification-guide.md).

## Fetch the ACI

The simplest way to retrieve the etcd ACI is to use image discovery:

```
./rkt fetch coreos.com/etcd:v2.0.9
rkt: searching for app image coreos.com/etcd:v2.0.9
rkt: fetching image from https://github.com/coreos/etcd/releases/download/v2.0.9/etcd-v2.0.9-linux-amd64.aci
Downloading signature from https://github.com/coreos/etcd/releases/download/v2.0.9/etcd-v2.0.9-linux-amd64.aci.asc
Downloading ACI: [================================             ] 2.71 MB/3.79 MB
rkt: signature verified:
  CoreOS ACI Builder <release@coreos.com>
  sha512-91e98d7f1679a097c878203c9659f2a2
```

For more on this and other ways to retrieve ACIs, check out the `rkt fetch` section of the [commands guide](https://github.com/coreos/rkt/blob/master/Documentation/commands.md#rkt-fetch).

## Run the ACI

Finally, let's run the application we just retrieved:

```
./rkt run coreos.com/etcd:v2.0.9
rkt: searching for app image coreos.com/etcd:v2.0.9
rkt: found image in local store, skipping fetching from https://github.com/coreos/etcd/releases/download/v2.0.9/etcd-v2.0.9-linux-amd64.aci
[  489.734930] etcd[4]: 2015/08/03 08:57:23 etcd: no data-dir provided, using default data-dir ./default.etcd
[  489.739297] etcd[4]: 2015/08/03 08:57:23 etcd: listening for peers on http://localhost:2380
[  489.740653] etcd[4]: 2015/08/03 08:57:23 etcd: listening for peers on http://localhost:7001
[  489.741405] etcd[4]: 2015/08/03 08:57:23 etcd: listening for client requests on http://localhost:2379
[  489.742178] etcd[4]: 2015/08/03 08:57:23 etcd: listening for client requests on http://localhost:4001
[  489.743394] etcd[4]: 2015/08/03 08:57:23 etcdserver: datadir is valid for the 2.0.1 format
[  489.743977] etcd[4]: 2015/08/03 08:57:23 etcdserver: name = default
[  489.744707] etcd[4]: 2015/08/03 08:57:23 etcdserver: data dir = default.etcd
[  489.745374] etcd[4]: 2015/08/03 08:57:23 etcdserver: member dir = default.etcd/member
[  489.746029] etcd[4]: 2015/08/03 08:57:23 etcdserver: heartbeat = 100ms
[  489.746688] etcd[4]: 2015/08/03 08:57:23 etcdserver: election = 1000ms
[  489.747238] etcd[4]: 2015/08/03 08:57:23 etcdserver: snapshot count = 10000
[  489.747586] etcd[4]: 2015/08/03 08:57:23 etcdserver: advertise client URLs = http://localhost:2379,http://localhost:4001
[  489.747838] etcd[4]: 2015/08/03 08:57:23 etcdserver: initial advertise peer URLs = http://localhost:2380,http://localhost:7001
[  489.748232] etcd[4]: 2015/08/03 08:57:23 etcdserver: initial cluster = default=http://localhost:2380,default=http://localhost:7001
[  489.750110] etcd[4]: 2015/08/03 08:57:23 etcdserver: start member ce2a822cea30bfca in cluster 7e27652122e8b2ae
[  489.750588] etcd[4]: 2015/08/03 08:57:23 raft: ce2a822cea30bfca became follower at term 0
[  489.750839] etcd[4]: 2015/08/03 08:57:23 raft: newRaft ce2a822cea30bfca [peers: [], term: 0, commit: 0, applied: 0, lastindex: 0, lastterm: 0]
[  489.751106] etcd[4]: 2015/08/03 08:57:23 raft: ce2a822cea30bfca became follower at term 1
[  489.751990] etcd[4]: 2015/08/03 08:57:23 etcdserver: added local member ce2a822cea30bfca [http://localhost:2380 http://localhost:7001] to cluster 7e27652122e8b2ae
[  491.050049] etcd[4]: 2015/08/03 08:57:25 raft: ce2a822cea30bfca is starting a new election at term 1
[  491.051266] etcd[4]: 2015/08/03 08:57:25 raft: ce2a822cea30bfca became candidate at term 2
[  491.052159] etcd[4]: 2015/08/03 08:57:25 raft: ce2a822cea30bfca received vote from ce2a822cea30bfca at term 2
[  491.053349] etcd[4]: 2015/08/03 08:57:25 raft: ce2a822cea30bfca became leader at term 2
[  491.053727] etcd[4]: 2015/08/03 08:57:25 raft.node: ce2a822cea30bfca elected leader ce2a822cea30bfca at term 2
[  491.054883] etcd[4]: 2015/08/03 08:57:25 etcdserver: published {Name:default ClientURLs:[http://localhost:2379 http://localhost:4001]} to cluster 7e27652122e8b2ae
```

Congratulations!
You've run your first application with rkt.
For more on how to use rkt, check out the [commands guide](https://github.com/coreos/rkt/blob/master/Documentation/commands.md).
