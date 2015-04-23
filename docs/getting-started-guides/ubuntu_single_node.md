## Getting started on Ubuntu 

This document describes how to get started to run kubernetes services on a single host (which is acting both as master and minion) for ubuntu systems. It consists of three steps

1. Make kubernetes and etcd binaries
2. Install upstart scripts
3. Customizing ubuntu launch

### 1. Make kubernetes and etcd binaries
Either build or download the latest [kubernetes binaries] (http://docs.k8s.io/getting-started-guides/binary_release.md)

Copy the kube binaries into `/opt/bin` or a path of your choice

Similarly pull an `etcd` binary from [etcd releases](https://github.com/coreos/etcd/releases) or build the `etcd` yourself using instructions at [https://github.com/coreos/etcd](https://github.com/coreos/etcd)

Copy the `etcd` binary into `/opt/bin` or path of your choice

### 2. Install upstart scripts
Running ubuntu/util.sh would install/copy the scripts for upstart to pick up. The script may warn you on some valid problems/conditions

```
$ cd kubernetes/cluster/ubuntu
$ sudo ./util.sh
```

After this the kubernetes and `etcd` services would be up and running. You can use `service start/stop/restart/force-reload` on the services.

Launching and scheduling containers using kubectl can also be used at this point, as explained mentioned in the [examples](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/guestbook)

### 3. Customizing the ubuntu launch
To customize the defaults you will need to tweak `/etc/default/kube*` files and restart the appropriate services. This is needed if the binaries are copied in a place other than `/opt/bin`. A run could look like

```
$ sudo cat /etc/default/etcd 
# Etcd Upstart and SysVinit configuration file

# Customize etcd location 
# ETCD="/opt/bin/etcd"

# Use ETCD_OPTS to modify the start/restart options
ETCD_OPTS="-listen-client-urls=http://127.0.0.1:4001"

# Add more environment settings used by etcd here

$ sudo service etcd status
etcd start/running, process 834
$ sudo service etcd restart
etcd stop/waiting
etcd start/running, process 29050
```
