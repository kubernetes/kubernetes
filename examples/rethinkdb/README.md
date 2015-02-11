RethinkDB Cluster on Kubernetes
==============================

Setting up a [rethinkdb](http://rethinkdb.com/) cluster on [kubernetes](http://kubernetes.io)

**Features**

 * Auto configuration cluster by querying info from k8s
 * Simple

Quick start
-----------

**Step 1**

antmanler/rethinkdb will discover peer using endpoints provided by kubernetes_ro service,
so first create a service so the following pod can query its endpoint

```shell
kubectl create -f driver-service.yaml
```

check out:

```shell
$kubectl get se
NAME                LABELS              SELECTOR                  IP                  PORT
rethinkdb-driver    db=influxdb         db=rethinkdb              10.241.105.47       28015
```

**Step 2**

start fist server in cluster

```shell
kubectl create -f rc.yaml
```

Actually, you can start servers as many as you want at one time, just modify the `replicas` in `rc.ymal`

check out again:

```shell
$kubectl get po
POD                                    IP                  CONTAINER(S)        IMAGE(S)            HOST                LABELS                       STATUS
99f6d361-abd6-11e4-a1ea-001c426dbc28   10.240.2.68         rethinkdb           rethinkdb:1.16.0    10.245.2.2/         db=rethinkdb,role=replicas   Running
```

**Done!**


---

Scale
-----

You can scale up you cluster using `kubectl resize`, and new pod will join to exsits cluster automatically, for example


```shell
$kubectl resize rc rethinkdb-rc-1.16.0 --replicas=3
resized
$kubectl get po
POD                                    IP                  CONTAINER(S)        IMAGE(S)            HOST                LABELS                       STATUS
99f6d361-abd6-11e4-a1ea-001c426dbc28   10.240.2.68         rethinkdb           rethinkdb:1.16.0    10.245.2.2/         db=rethinkdb,role=replicas   Running
d10182b5-abd6-11e4-a1ea-001c426dbc28   10.240.26.14        rethinkdb           rethinkdb:1.16.0    10.245.2.4/         db=rethinkdb,role=replicas   Running
d101c1a4-abd6-11e4-a1ea-001c426dbc28   10.240.11.14        rethinkdb           rethinkdb:1.16.0    10.245.2.3/         db=rethinkdb,role=replicas   Running
```

Admin
-----

You need a separate pod (which labled as role:admin) to access Web Admin UI

```shell
kubectl create -f admin-pod.yaml
kubectl create -f admin-service.yaml
```

find the service

```shell
$kubectl get se
NAME                LABELS              SELECTOR                  IP                  PORT
rethinkdb-admin     db=influxdb         db=rethinkdb,role=admin   10.241.220.209      8080
rethinkdb-driver    db=influxdb         db=rethinkdb              10.241.105.47       28015
```

open a web browser and access to *http://10.241.220.209:8080* to manage you cluster

**Why not just using pods in replicas?**

This is because kube-proxy will act as a load balancer and send your traffic to different server,
since the ui is not stateless when playing with Web Admin UI will cause `Connection not open on server` error.


- - -

**BTW**

  * All services and pods are placed under namespace `rethinkdb`.

  * `gen_pod.sh` is using to generate pod templates for my local cluster,
the generated pods which is using `nodeSelector` to force k8s to schedule containers to my designate nodes, for I need to access persistent data on my host dirs.

  * see [antmanler/rethinkdb-k8s](https://github.com/antmanler/rethinkdb-k8s) for detail
