---
page_title: Docker discovery
page_description: discovery
page_keywords: docker, clustering, discovery
---

# Discovery

Docker comes with multiple Discovery backends.

## Backends

### Using etcd

Point your Docker Engine instances to a common etcd instance. You can specify
the address Docker uses to advertise the node using the `--cluster-advertise`
flag.

```bash
$ dockerd -H=<node_ip:2376> --cluster-advertise=<node_ip:2376> --cluster-store etcd://<etcd_ip1>,<etcd_ip2>/<path>
```

### Using consul

Point your Docker Engine instances to a common Consul instance. You can specify
the address Docker uses to advertise the node using the `--cluster-advertise`
flag.

```bash
$ dockerd -H=<node_ip:2376> --cluster-advertise=<node_ip:2376> --cluster-store consul://<consul_ip>/<path>
```

### Using zookeeper

Point your Docker Engine instances to a common Zookeeper instance. You can specify
the address Docker uses to advertise the node using the `--cluster-advertise`
flag.

```bash
$ dockerd -H=<node_ip:2376> --cluster-advertise=<node_ip:2376> --cluster-store zk://<zk_addr1>,<zk_addr2>/<path>
```
