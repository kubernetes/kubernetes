<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Example of NFS volume

See [nfs-web-rc.yaml](nfs-web-rc.yaml) for a quick example of how to use an NFS
volume claim in a replication controller. It relies on the
[NFS persistent volume](nfs-pv.yaml) and
[NFS persistent volume claim](nfs-pvc.yaml) in this example as well.

## Complete setup

The example below shows how to export a NFS share from a single pod replication
controller and import it into two replication controllers.

### NFS server part

Define [NFS server controller](nfs-server-rc.yaml) and
[NFS service](nfs-server-service.yaml):

```console
$ kubectl create -f examples/nfs/nfs-server-rc.yaml
$ kubectl create -f examples/nfs/nfs-server-service.yaml
```

The server exports `/mnt/data` directory as `/` (fsid=0). The
directory contains dummy `index.html`. Wait until the pod is running
by checking `kubectl get pods -lrole=nfs-server`.

### Create the NFS claim

The [NFS busybox controller](nfs-busybox-rc.yaml) uses a simple script to
generate data written to the NFS server we just started. First, you'll need to
find the cluster IP of the server:

```console
$ kubectl describe services nfs-server
```

Replace the invalid IP in the [nfs PV](nfs-pv.yaml). (In the future,
we'll be able to tie these together using the service names, but for
now, you have to hardcode the IP.)

Create the the [persistent volume](../../docs/user-guide/persistent-volumes.md)
and the persistent volume claim for your NFS server. The persistent volume and
claim gives us an indirection that allow multiple pods to refer to the NFS
server using a symbolic name rather than the hardcoded server address.

```console
$ kubectl create -f examples/nfs/nfs-pv.yaml
$ kubectl create -f examples/nfs/nfs-pvc.yaml
```

## Setup the fake backend

The [NFS busybox controller](nfs-busybox-rc.yaml) updates `index.html` on the
NFS server every 10 seconds. Let's start that now:

```console
$ kubectl create -f examples/nfs/nfs-busybox-rc.yaml
```

Conveniently, it's also a `busybox` pod, so we can get an early check
that our mounts are working now. Find a busybox pod and exec:

```console
$ kubectl get pod -lname=nfs-busybox
NAME                READY     STATUS    RESTARTS   AGE
nfs-busybox-jdhf3   1/1       Running   0          25m
nfs-busybox-w3s4t   1/1       Running   0          25m
$ kubectl exec nfs-busybox-jdhf3 -- cat /mnt/index.html
Thu Oct 22 19:20:18 UTC 2015
nfs-busybox-w3s4t
```

You should see output similar to the above if everything is working well. If
it's not, make sure you changed the invalid IP in the [NFS PV](nfs-pv.yaml) file
and make sure the `describe services` command above had endpoints listed
(indicating the service was associated with a running pod).

### Setup the web server

The [web server controller](nfs-web-rc.yaml) is an another simple replication
controller demonstrates reading from the NFS share exported above as a NFS
volume and runs a simple web server on it.

Define the pod:

```console
$ kubectl create -f examples/nfs/nfs-web-rc.yaml
```

This creates two pods, each of which serve the `index.html` from above. We can
then use a simple service to front it:

```console
kubectl create -f examples/nfs/nfs-web-service.yaml
```

We can then use the busybox container we launched before to check that `nginx`
is serving the data appropriately:

```console
$ kubectl get pod -lname=nfs-busybox
NAME                READY     STATUS    RESTARTS   AGE
nfs-busybox-jdhf3   1/1       Running   0          1h
nfs-busybox-w3s4t   1/1       Running   0          1h
$ kubectl get services nfs-web
NAME      LABELS    SELECTOR            IP(S)        PORT(S)
nfs-web   <none>    role=web-frontend   10.0.68.37   80/TCP
$ kubectl exec nfs-busybox-jdhf3 -- wget -qO- http://10.0.68.37
Thu Oct 22 19:28:55 UTC 2015
nfs-busybox-w3s4t
```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/nfs/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
