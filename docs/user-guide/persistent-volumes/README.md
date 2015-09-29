<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/persistent-volumes/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# How To Use Persistent Volumes

The purpose of this guide is to help you become familiar with [Kubernetes Persistent Volumes](../persistent-volumes.md).  By the end of the guide, we'll have
nginx serving content from your persistent volume.

This guide assumes knowledge of Kubernetes fundamentals and that you have a cluster up and running.

See [Persistent Storage design document](../../design/persistent-storage.md) for more information.

## Provisioning

A Persistent Volume (PV) in Kubernetes represents a real piece of underlying storage capacity in the infrastructure.  Cluster administrators
must first create storage (create their Google Compute Engine (GCE) disks, export their NFS shares, etc.) in order for Kubernetes to mount it.

PVs are intended for "network volumes" like GCE Persistent Disks, NFS shares, and AWS ElasticBlockStore volumes.  `HostPath` was included
for ease of development and testing.  You'll create a local `HostPath` for this example.

> IMPORTANT! For `HostPath` to work, you will need to run a single node cluster.  Kubernetes does not
support local storage on the host at this time.  There is no guarantee your pod ends up on the correct node where the `HostPath` resides.



```console
# This will be nginx's webroot
$ mkdir /tmp/data01
$ echo 'I love Kubernetes storage!' > /tmp/data01/index.html
```

PVs are created by posting them to the API server.

```console
$ kubectl create -f docs/user-guide/persistent-volumes/volumes/local-01.yaml
NAME      LABELS       CAPACITY      ACCESSMODES   STATUS      CLAIM     REASON
pv0001    type=local   10737418240   RWO           Available 
```

## Requesting storage

Users of Kubernetes request persistent storage for their pods.  They don't know how the underlying cluster is provisioned.
They just know they can rely on their claim to storage and can manage its lifecycle independently from the many pods that may use it.

Claims must be created in the same namespace as the pods that use them.

```console
$ kubectl create -f docs/user-guide/persistent-volumes/claims/claim-01.yaml

$ kubectl get pvc
NAME                LABELS              STATUS              VOLUME
myclaim-1           map[]                                   
           
           
# A background process will attempt to match this claim to a volume.
# The eventual state of your claim will look something like this:

$ kubectl get pvc
NAME        LABELS    STATUS    VOLUME
myclaim-1   map[]     Bound     pv0001

$ kubectl get pv
NAME      LABELS       CAPACITY      ACCESSMODES   STATUS    CLAIM               REASON
pv0001    type=local   10737418240   RWO           Bound     default/myclaim-1 
```

## Using your claim as a volume

Claims are used as volumes in pods.  Kubernetes uses the claim to look up its bound PV.  The PV is then exposed to the pod.

```console
$ kubectl create -f docs/user-guide/persistent-volumes/simpletest/pod.yaml

$ kubectl get pods
NAME      READY     STATUS    RESTARTS   AGE
mypod     1/1       Running   0          1h

$ kubectl create -f docs/user-guide/persistent-volumes/simpletest/service.json
$ kubectl get services
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR           AGE
frontendservice   10.0.0.241       <none>            3000/TCP      name=frontendhttp  1d
kubernetes        10.0.0.2         <none>            443/TCP       <none>             2d
```

## Next steps

You should be able to query your service endpoint and see what content nginx is serving.  A "forbidden" error might mean you
need to disable SELinux (setenforce 0).

```console
$ curl 10.0.0.241:3000
I love Kubernetes storage!
```

Hopefully this simple guide is enough to get you started with PersistentVolumes.  If you have any questions, join the team on [Slack](../../troubleshooting.md#slack) and ask!

Enjoy!


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/persistent-volumes/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
