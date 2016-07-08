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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Persistent Volumes with Kubernetes + libStorage

### Pre-Reqisites

* Kubernetes 1.3 or above
* RexRay 0.4.0 or above

### Case 1: PersistentVolumes Bound to Pre-Defined Volume

**Define RexRay Volume**

```
#> rexray volume create --volumename="vol-0001" --size=1
attachments: []
availabilityzone: ""
iops: 0
name: vol-0001
networkname: ""
size: 1
status: ""
id: af76dab6-ba1c-4788-bdb8-9f63e0cd62db
type: HardDisk
fields: {}
```

**Persistent Volume**

```
kind: PersistentVolume
apiVersion: v1
metadata:
  name: pv-0001
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  libstorage:
    volumeName: vol-0001
    service: kubernetes
```

**Persistent Volume Claim**

```
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: pvc-0001
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

**Result of Deployment**

```
get pv
NAME      CAPACITY   ACCESSMODES   STATUS    CLAIM              REASON    AGE
pv-0001   1Gi        RWO           Bound     default/pvc-0001             14m
```

**Deploy Pod with Claim**

```
kind: Pod
apiVersion: v1
metadata:
  name: pod0002
spec:
  containers:
    - name: pod0002-container
      image: gcr.io/google_containers/test-webserver
      volumeMounts:
      - mountPath: /test
        name: test-data
  volumes:
    - name: test-data
      persistentVolumeClaim:
        claimName: pvc-0001
```

**Result**
`cluster/kubectl.sh describe pod pod0002` shows pod info including volume information (see below).

```
. . .
Conditions:
  Type		Status
  Initialized 	True
  Ready 	True
  PodScheduled 	True
Volumes:
  test-data:
    Type:	PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:	pvc-0001
    ReadOnly:	false
. . .
```

## Case 2: Dynamic Persistent Volume Provisioner

In this scenario, the volume is defined in one Kubernetes claim file.  The file uses the experimental annotation `volume.alpha.kubernetes.io/storage-class` to indicate it wants the volume to be dynamically be provisioned.  Then the libStorage Volume provisioner will create the volume using RexRay automatically.

### Activating LibStorage Provisioning

**New Kube-Controller-Manager Flag Added**
The code now supports a new flag `--enable-libstorage-provisioner`, for the controller binary `kube-controller-manager`, that activates the libStorage persistent volume provisioner to handle the automatic provisioning of the volume defined in a PersistentVolumeClaim yaml.

**Launching Local Cluster with Provisioner**
The following activates libStorage provisioning flag in the `hack/local-up-cluster.sh` script that comes with Kubernetes.

```
KUBERNETES_PROVIDER=local ENABLE_LIBSTORAGE_PROVISIONER=true LOG_LEVEL=99 hack/local-up-cluster.sh
```

The following snippet shows how the `kube-controller-manager` is launched in the bash script wit the new flag:

```
sudo -E "${GO_OUT}/hyperkube" controller-manager \
      --v=${LOG_LEVEL} \
...
      --enable-libstorage-provisioner="${ENABLE_LIBSTORAGE_PROVISIONER}" \
...
```

### Deploy a Persistent Volume Claim

**Define PersistentVolumeClaim**

```
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: pvc-0002
  annotations:
    volume.experimental.kubernetes.io/provisioning-required: "true"
    volume.alpha.kubernetes.io/storage-class: kubernetes
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

**Description of PVC Created**

```
#> cluster/kubectl.sh describe pvc pvc-0002
Name:		pvc-0002
Namespace:	default
Status:		Bound
Volume:		pvc-850a3832-3548-11e6-ba16-0800275f7ed0
Labels:		<none>
Capacity:	0
Access Modes:
No events.
```

**PersistentVolume Description**
Notice there is a volume ID reported (awesome!)

```
#> cluster/kubectl.sh describe pv pvc-850a3832-3548-11e6-ba16-0800275f7ed0
Name:		pvc-850a3832-3548-11e6-ba16-0800275f7ed0
Labels:		<none>
Status:		Bound
Claim:		default/pvc-0002
Reclaim Policy:	Delete
Access Modes:	RWO
Capacity:	1Gi
Message:
Source:
    Type:	LibStorage (a persistent disk resource in libStorage)
    VolumeName:	kubernetes-dynamic-pvc-850a3832-3548-11e6-ba16-0800275f7ed0
    VolumeID:	6699738b-17d1-41a3-8cce-b390dcab09ed
```

**Validate Volume with RexRay**

```
#> rexray volume get --volumeid=6699738b-17d1-41a3-8cce-b390dcab09ed
attachments: []
availabilityzone: ""
iops: 0
name: kubernetes-dynamic-pvc-850a3832-3548-11e6-ba16-0800275f7ed0
networkname: ""
size: 1
status: /Users/vladimir/VirtualBox Volumes/kubernetes-dynamic-pvc-850a3832-3548-11e6-ba16-0800275f7ed0
id: 6699738b-17d1-41a3-8cce-b390dcab09ed
type: ""
fields: {}
```

**Launch Pod Using PVC**
We can launch a POD that uses the claim defined above.

```
kind: Pod
apiVersion: v1
metadata:
  name: pod0003
spec:
  containers:
    - name: pod0003-container
      image: gcr.io/google_containers/test-webserver
      volumeMounts:
      - mountPath: /test
        name: test-data
  volumes:
    - name: test-data
      persistentVolumeClaim:
        claimName: pvc-0002
```

**PVC Bound**
You can see now pvc-0002 is now bound.

```
#> cluster/kubectl.sh describe pvc pvc-0002
Name:		pvc-0002
Namespace:	default
Status:		Bound
Volume:		pvc-850a3832-3548-11e6-ba16-0800275f7ed0
...
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/libstorage/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
