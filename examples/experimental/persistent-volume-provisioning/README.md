<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/examples/experimental/persistent-volume-provisioning/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Persistent Volume Provisioning

This example shows how to use experimental persistent volume provisioning.

### Pre-requisites

This example assumes that you have an understanding of Kubernetes administration and can modify the
scripts that launch kube-controller-manager.

### Admin Configuration

The admin must define `StorageClass` objects that describe named "classes" of storage offered in a cluster. Different classes might map to arbitrary levels or policies determined by the admin. When configuring a `StorageClass` object for persistent volume provisioning, the admin will need to describe the type of provisioner to use and the parameters that will be used by the provisioner when it provisions a `PersistentVolume` belonging to the class.

The name of a StorageClass object is significant, and is how users can request a particular class, by specifying the name in their `PersistentVolumeClaim`. The `provisioner` field must be specified as it determines what volume plugin is used for provisioning PVs. 2 cloud providers will be provided in the beta version of this feature: EBS and GCE. The `parameters` field contains the parameters that describe volumes belonging to the storage class. Different parameters may be accepted depending on the `provisioner`. For example, the value `io1`, for the parameter `type`, and the parameter `iopsPerGB` are specific to EBS . When a parameter is omitted, some default is used.

#### AWS

```yaml
kind: StorageClass
apiVersion: extensions/v1beta1
metadata:
  name: slow
provisioner: kubernetes.io/aws-ebs
parameters:
  type: io1
  zone: us-east-1d
  iopsPerGB: "10"
```

* `type`: `io1`, `gp2`, `sc1`, `st1`. See AWS docs for details. Default: `gp2`.
* `zone`: AWS zone
* `iopsPerGB`: only for `io1` volumes. I/O operations per second per GiB. AWS volume plugin multiplies this with size of requested volume to compute IOPS of the volume and caps it at 20 000 IOPS (maximum supported by AWS, see AWS docs).

#### GCE

```yaml
kind: StorageClass
apiVersion: extensions/v1beta1
metadata:
  name: slow
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-standard
  zone: us-central1-a
```

* `type`: `pd-standard` or `pd-ssd`. Default: `pd-ssd`
* `zone`: GCE zone

### User provisioning requests

Users request dynamically provisioned storage by including a storage class in their `PersistentVolumeClaim`.
The annotation `volume.beta.kubernetes.io/storage-class` is used to access this experimental feature. It is required that this value matches the name of a `StorageClass` configured by the administrator.
In the future, the storage class may remain in an annotation or become a field on the claim itself.

```
{
  "kind": "PersistentVolumeClaim",
  "apiVersion": "v1",
  "metadata": {
    "name": "claim1",
    "annotations": {
        "volume.beta.kubernetes.io/storage-class": "slow"
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

This example uses gce but any provisioner would follow the same flow.

First we note there are no Persistent Volumes in the cluster.  After creating a storage class and a claim including that storage class, we see a new PV is created
and automatically bound to the claim requesting storage.


``` 
$ kubectl get pv

$ kubectl create -f examples/experimental/persistent-volume-provisioning/gce-pd.yaml
storageclass "slow" created

$ kubectl create -f examples/experimental/persistent-volume-provisioning/claim1.json
persistentvolumeclaim "claim1" created

$ kubectl get pv
NAME                                       CAPACITY   ACCESSMODES   STATUS    CLAIM                        REASON    AGE
pvc-bb6d2f0c-534c-11e6-9348-42010af00002   3Gi        RWO           Bound     default/claim1                         4s

$ kubectl get pvc
NAME      LABELS    STATUS    VOLUME                                     CAPACITY   ACCESSMODES   AGE
claim1    <none>    Bound     pvc-bb6d2f0c-534c-11e6-9348-42010af00002   3Gi        RWO           7s

# delete the claim to release the volume
$ kubectl delete pvc claim1
persistentvolumeclaim "claim1" deleted

# the volume is deleted in response to being release of its claim
$ kubectl get pv

```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/experimental/persistent-volume-provisioning/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
