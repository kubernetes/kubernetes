# Flannel integration with Kubernetes

## Why?

* Networking works out of the box.
* Cloud gateway configuration is regulated.
* Consistent bare metal and cloud experience.
* Lays foundation for integrating with networking backends and vendors.

# How?

```
Master                      Node1
---------------------|--------------------------------
database             |
    |                |
{10.250.0.0/16}      |         docker
    |         here's podcidr    |restart with podcidr
apiserver <------------------- kubelet
    |                |          |here's podcidr
flannel-server:10253 <------- flannel-daemon
                     --/16--->
                     <--watch--  [config iptables]
                subscribe to new node subnets
                      -------->   [config VXLan]
                     |
```

There is a tiny lie in the above diagram, as of now, the flannel server on the master maintains a private etcd. This will not be necessary once we have a generalized network resource, and a Kubernetes x flannel backend.

# Limitations

* Integration is experimental

# Wishlist
