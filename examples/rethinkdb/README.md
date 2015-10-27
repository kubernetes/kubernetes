<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
RethinkDB Cluster on Kubernetes
==============================

Setting up a [rethinkdb](http://rethinkdb.com/) cluster on [kubernetes](http://kubernetes.io)

**Features**

 * Auto configuration cluster by querying info from k8s
 * Simple

Quick start
-----------

**Step 1**

Rethinkdb will discover its peer using endpoints provided by kubernetes service,
so first create a service so the following pod can query its endpoint

```sh
$kubectl create -f examples/rethinkdb/driver-service.yaml
```

check out:

```sh
$kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
rethinkdb-driver  10.0.27.114      <none>            28015/TCP     db=rethinkdb           10m
[...]
```

**Step 2**

start the first server in the cluster

```sh
$kubectl create -f examples/rethinkdb/rc.yaml
```

Actually, you can start servers as many as you want at one time, just modify the `replicas` in `rc.ymal`

check out again:

```sh
$kubectl get pods
NAME                                                  READY     REASON    RESTARTS   AGE
[...]
rethinkdb-rc-r4tb0                                    1/1       Running   0          1m
```

**Done!**


---

Scale
-----

You can scale up your cluster using `kubectl scale`. The new pod will join to the existing cluster automatically, for example


```sh
$kubectl scale rc rethinkdb-rc --replicas=3
scaled

$kubectl get pods
NAME                                                  READY     REASON    RESTARTS   AGE
[...]
rethinkdb-rc-f32c5                                    1/1       Running   0          1m
rethinkdb-rc-m4d50                                    1/1       Running   0          1m
rethinkdb-rc-r4tb0                                    1/1       Running   0          3m
```

Admin
-----

You need a separate pod (labeled as role:admin) to access Web Admin UI

```sh
kubectl create -f examples/rethinkdb/admin-pod.yaml
kubectl create -f examples/rethinkdb/admin-service.yaml
```

find the service

```console
$kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR                  AGE
[...]
rethinkdb-admin   10.0.131.19      104.197.19.120    8080/TCP      db=rethinkdb,role=admin   10m
rethinkdb-driver  10.0.27.114      <none>            28015/TCP     db=rethinkdb              20m
```

We request an external load balancer in the [admin-service.yaml](admin-service.yaml) file:

```
type: LoadBalancer
```

The external load balancer allows us to access the service from outside the firewall via an external IP, 104.197.19.120 in this case.

Note that you may need to create a firewall rule to allow the traffic, assuming you are using Google Compute Engine:

```console
$ gcloud compute firewall-rules create rethinkdb --allow=tcp:8080
```

Now you can open a web browser and access to *http://104.197.19.120:8080* to manage your cluster.



**Why not just using pods in replicas?**

This is because kube-proxy will act as a load balancer and send your traffic to different server,
since the ui is not stateless when playing with Web Admin UI will cause `Connection not open on server` error.


- - -

**BTW**

  * `gen_pod.sh` is using to generate pod templates for my local cluster,
the generated pods which is using `nodeSelector` to force k8s to schedule containers to my designate nodes, for I need to access persistent data on my host dirs. Note that one needs to label the node before 'nodeSelector' can work, see this [tutorial](../../docs/user-guide/node-selection/)

  * see [antmanler/rethinkdb-k8s](https://github.com/antmanler/rethinkdb-k8s) for detail




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/rethinkdb/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
