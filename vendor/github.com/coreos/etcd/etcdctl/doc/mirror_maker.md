## Mirror Maker

Mirror maker mirrors a prefix in the key-value space of an etcd cluster into another prefix in another cluster. Mirroring is designed for copying configuration to various clusters distributed around the world. Mirroring usually has very low latency once it completes synchronizing with the initial state. Mirror maker utilizes the etcd watcher facility to immediately inform the mirror of any key modifications. Based on our experiments, the network latency between the mirror maker and the two clusters accounts for most of the latency. If the network is healthy, copying configuration held in etcd to the mirror should take under one second even for a world-wide deployment.

If the mirror maker fails to connect to one of the clusters, the mirroring will pause. Mirroring can  be resumed automatically once connectivity is reestablished.

The mirroring mechanism is unidirectional. Data under the destination cluster’s mirroring prefix should be treated as read only. The mirror maker only mirrors key-value pairs; metadata, such as version number or modification revision, is discarded. However, mirror maker still attempts to preserve update ordering during normal operation, but there is no ordering guarantee during initial sync nor during failure recovery following network interruption. As a rule of thumb, the ordering of the updates on the mirror should not be considered reliable.

```
+-------------+
|             |
|  source     |      +-----------+
|  cluster    +----> |  mirror   |
|             |      |  maker    |
+-------------+      +---+-------+
                         |
                         v
               +-------------+
               |             |
               |    mirror  |
               |    cluster  |
               |             |
               +-------------+

```

Mirror-maker is a built-in feature of [etcdctl][etcdctl].

[etcdctl]: ../README.md
