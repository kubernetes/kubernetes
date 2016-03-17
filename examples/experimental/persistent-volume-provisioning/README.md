<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Persistent Volume Provisioning

This example shows how to use experimental persistent volume provisioning.

### Pre-requisites

This example assumes that you have an understanding of Kubernetes administration and can modify the
scripts that launch kube-controller-manager.

### Admin Configuration

No configuration is required by the admin!  3 cloud providers will be provided in the alpha version
of this feature: EBS, GCE, and Cinder.

When Kubernetes is running in one of those clouds, there will be an implied provisioner.
There is no provisioner when running outside of any of those 3 cloud providers.

A fourth provisioner is included for testing and development only.  It creates HostPath volumes,
which will never work outside of a single node cluster. It is not supported in any way except for
local for testing and development.


### User provisioning requests

Users request dynamically provisioned storage by including a storage class in their `PersistentVolumeClaim`.
The annotation `volume.alpha.kubernetes.io/storage-class` is used to access this experimental feature.
In the future, admins will be able to define many storage classes.
The storage class may remain in an annotation or become a field on the claim itself.

> The value of the storage-class annotation does not matter in the alpha version of this feature.  There is
a single implied provisioner per cloud (which creates 1 kind of volume in the provider).  The full version of the feature
will require that this value matches what is configured by the administrator.

```
{
  "kind": "PersistentVolumeClaim",
  "apiVersion": "v1",
  "metadata": {
    "name": "claim1",
    "annotations": {
        "volume.alpha.kubernetes.io/storage-class": "foo"
    }
  },
  "spec": {
    "accessModes": [
      "ReadWriteOnce"
    ],
    "resources": {
      "requests": {
        "storage": "3Gi"
      }
    }
  }
}
```

### Sample output

This example uses HostPath but any provisioner would follow the same flow.

First we note there are no Persistent Volumes in the cluster.  After creating a claim, we see a new PV is created
and automatically bound to the claim requesting storage.


``` 
$ kubectl get pv

$ kubectl create -f examples/experimental/persistent-volume-provisioning/claim1.json
I1012 13:07:57.666759   22875 decoder.go:141] decoding stream as JSON
persistentvolumeclaim "claim1" created

$ kubectl get pv
NAME                LABELS                                   CAPACITY   ACCESSMODES   STATUS    CLAIM            REASON    AGE
pv-hostpath-r6z5o   createdby=hostpath-dynamic-provisioner   3Gi        RWO           Bound     default/claim1             2s

$ kubectl get pvc
NAME      LABELS    STATUS    VOLUME              CAPACITY   ACCESSMODES   AGE
claim1    <none>    Bound     pv-hostpath-r6z5o   3Gi        RWO           7s

# delete the claim to release the volume
$ kubectl delete pvc claim1
persistentvolumeclaim "claim1" deleted

# the volume is deleted in response to being release of its claim
$ kubectl get pv

```


### Provision GlusterFS Volume
Enable glusterfs dynamic provisioning by supplying `--enable-network-storage-provisioner=true --storage-config=/path/to/storage/config/directory` to `kube-controller-manager`.

In the storage configuration directory, provide a gluster cluster configuration json like the following, where `endpoint` and `resturl` are mandatory.
```gluster.json
{
   "endpoint": "glusterfs-cluster",
   "resturl": "http://127.0.0.1:8081",
   "restauthenabled":false,
   "restuser":"",
   "restuserkey":""
}

```
Create a PVC json like the following:

```json
{
  "kind": "PersistentVolumeClaim",
  "apiVersion": "v1",
  "metadata": {
    "name": "glusterc",
    "annotations": {
      "volume.alpha.kubernetes.io/storage-class": "foo"
    }
  },
  "spec": {
    "accessModes": [
      "ReadOnlyMany"
    ],
    "resources": {
      "requests": {
        "storage": "20Gi"
      }
    }
  }
}

```

Create the PVC using `kubectl`:

```console
# kubectl create -f claim1.json
persistentvolumeclaim "glusterc" created
```

Confirm the PV is dynamically created using glusterfs volume with the specified size:
```console
[root@node]# ./kubectl get pv,pvc
NAME                 CAPACITY   ACCESSMODES          STATUS     CLAIM              REASON    AGE
pv-glusterfs-921zw   20Gi       ROX                  Bound      default/glusterc             9s
NAME                 STATUS     VOLUME               CAPACITY   ACCESSMODES        AGE
glusterc             Bound      pv-glusterfs-921zw   20Gi       ROX                9s

[root@node]# ./kubectl describe pv pv-glusterfs-921zw
Name:		pv-glusterfs-921zw
Labels:		<none>
Status:		Bound
Claim:		default/glusterc
Reclaim Policy:	Delete
Access Modes:	ROX
Capacity:	20Gi
Message:
Source:
    Type:		Glusterfs (a Glusterfs mount on the host that shares a pod's lifetime)
    EndpointsName:	glusterfs-cluster
    Path:		vol_40045becbd6a42c83dfc9d3f5410b2f8
    ReadOnly:		false


```


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/experimental/persistent-volume-provisioning/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
