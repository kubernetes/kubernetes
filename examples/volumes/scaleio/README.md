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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Dell EMC ScaleIO Volume Plugin for Kubernetes

This document shows how to configure Kubernetes resources to consume storage from volumes hosted on ScaleIO cluster.

## Pre-Requisites

* Kubernetes ver 1.6 or later
* ScaleIO ver 2.0 or later
* A ScaleIO cluster with an API gateway
* ScaleIO SDC binary installed/configured on each Kubernetes node that will consume storage

## ScaleIO Setup

This document assumes you are familiar with ScaleIO and have a cluster ready to go.  If you are *not familiar* with ScaleIO, please review *Learn how to setup a 3-node* [ScaleIO cluster on Vagrant](https://github.com/codedellemc/labs/tree/master/setup-scaleio-vagrant) and see *General instructions on* [setting up ScaleIO](https://www.emc.com/products-solutions/trial-software-download/scaleio.htm)

For this demonstration, ensure the following: 

 - The ScaleIO `SDC` component is installed and properly configured on all Kubernetes nodes where deployed pods will consume ScaleIO-backed volumes.
 - You have a configured ScaleIO gateway that is accessible from the Kubernetes nodes. 

## Deploy Kubernetes Secret for ScaleIO

The ScaleIO plugin uses a Kubernetes Secret object to store the `username` and `password` credentials.  
Kuberenetes requires the secret values to be base64-encoded to simply obfuscate (not encrypt) the clear text as shown below.

```
$> echo -n "siouser" | base64
c2lvdXNlcg==
$> echo -n "sc@l3I0" | base64
c2NAbDNJMA==
```
The previous will generate `base64-encoded` values for the username and password.  
Remember to generate the credentials for your own environment and copy them in a secret file similar to the following.  

File: [secret.yaml](secret.yaml)

```
apiVersion: v1
kind: Secret
metadata:
  name: sio-secret
type: kubernetes.io/scaleio
data:
  username: c2lvdXNlcg==
  password: c2NAbDNJMA==
```

Notice the name of the secret specified above as `sio-secret`.  It will be referred in other YAML files.  Next, deploy the secret.

```
$ kubectl create -f ./examples/volumes/scaleio/secret.yaml
```

## Deploying Pods with Persistent Volumes

The example presented in this section shows how the ScaleIO volume plugin can automatically attach, format, and mount an existing ScaleIO volume for pod. 
The Kubernetes ScaleIO volume spec supports the following attributes:

| Attribute | Description |
|-----------|-------------|
| gateway | address to a ScaleIO API gateway (required)|
| system  | the name of the ScaleIO system (required)|
| protectionDomain| the name of the ScaleIO protection domain (default `default`)|
| storagePool| the name of the volume storage pool (default `default`)|
| storageMode| the storage provision mode: `ThinProvisionned` (default) or `ThickProvisionned`|
| volumeName| the name of an existing volume in ScaleIO (required)|
| secretRef:name| reference to a configured Secret object (required, see Secret earlier)|
| readOnly| specifies the access mode to the mounted volume (default `false`)|
| fsType| the file system to use for the volume (default `ext4`)|

### Create Volume

Static persistent volumes require that the volume, to be consumed by the pod, be already created in ScaleIO.  You can use your ScaleIO tooling to create a new volume or use the name of a volume that already exists in ScaleIO.  For this demo, we assume there's a volume named `vol-0`.  If you want to use an existing volume, ensure its name is reflected properly in the `volumeName` attribute below.

### Deploy Pod YAML

Create a pod YAML file that declares the volume (above) to be used.

File: [pod.yaml](pod.yaml)

```
apiVersion: v1
kind: Pod
metadata:
  name: pod-0
spec:
  containers:
  - image: gcr.io/google_containers/test-webserver
    name: pod-0
    volumeMounts:
    - mountPath: /test-pd
      name: vol-0
  volumes:
  - name: vol-0
    scaleIO:
      gateway: https://localhost:443/api
      system: scaleio
      volumeName: vol-0
      secretRef:
        name: sio-secret
      fsType: xfs
```
Notice the followings in the previous YAML:

- Update the `gatewway` to point to your ScaleIO gateway endpoint.
- The `volumeName` attribute refers to the name of an existing volume in ScaleIO.
- The `secretRef:name` attribute references the name of the secret object deployed earlier.

Next, deploy the pod.

```
$> kubectl create -f examples/volumes/scaleio/pod.yaml
```
You can verify the pod:
```
$> kubectl get pod
NAME      READY     STATUS    RESTARTS   AGE
pod-0     1/1       Running   0          33s
```
Or for more detail, use 
```
kubectl describe pod pod-0
```
You can see the attached/mapped volume on the node:
```
$> lsblk
NAME        MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
...
scinia      252:0    0    8G  0 disk /var/lib/kubelet/pods/135986c7-dcb7-11e6-9fbf-080027c990a7/volumes/kubernetes.io~scaleio/vol-0
```

## StorageClass and Dynamic Provisioning

In the example in this section, we will see how the ScaleIO volume plugin can automatically provision described in a `StorageClass`.
The ScaleIO volume plugin is a dynamic provisioner identified as `kubernetes.io/scaleio` and supports the following parameters:

| Parameter | Description |
|-----------|-------------|
| gateway | address to a ScaleIO API gateway (required)|
| system  | the name of the ScaleIO system (required)|
| protectionDomain| the name of the ScaleIO protection domain (default `default`)|
| storagePool| the name of the volume storage pool (default `default`)|
| storageMode| the storage provision mode: `ThinProvisionned` (default) or `ThickProvisionned`|
| secretRef| reference to the name of a configured Secret object (required)|
| readOnly| specifies the access mode to the mounted volume (default `false`)|
| fsType| the file system to use for the volume (default `ext4`)|


### ScaleIO StorageClass

Define a new `StorageClass` as shown in the following YAML.

File [sc.yaml](sc.yaml)

```
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: sio-small
provisioner: kubernetes.io/scaleio
parameters:
  gateway: https://localhost:443/api
  system: scaleio
  protectionDomain: default
  secretRef: sio-secret
  fsType: xfs
```
Note the followings:

- The `name` attribute is set to sio-small . It will be referenced later.
- The `secretRef` attribute matches the name of the Secret object created earlier.

Next, deploy the storage class file.

```
$> kubectl create -f examples/volumes/scaleio/sc.yaml

$> kubectl get sc
NAME        TYPE
sio-small   kubernetes.io/scaleio
```

### PVC for the StorageClass

The next step is to define/deploy a `PersistentVolumeClaim` that will use the StorageClass.

File [sc-pvc.yaml](sc-pvc.yaml)

```
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: pvc-sio-small
  annotations:
      volume.beta.kubernetes.io/storage-class: sio-small
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

Note the `annotations:` entry which specifies annotation `volume.beta.kubernetes.io/storage-class: sio-small` which references the name of the storage class defined earlier.

Next, we deploy PVC file for the storage class.  This step will cause the Kubernetes ScaleIO plugin to create the volume in the storage system.  
```
$> kubectl create -f examples/volumes/scaleio/sc-pvc.yaml
```
You verify that a new volume created in the ScaleIO dashboard.  You can also verify the newly created volume as follows.
```
 kubectl get pvc
NAME            STATUS    VOLUME                                     CAPACITY   ACCESSMODES   AGE
pvc-sio-small   Bound     pvc-5fc78518-dcae-11e6-a263-080027c990a7   10Gi       RWO           1h
```

###Pod for PVC and SC
At this point, the volume is created (by the claim) in the storage system.  To use it, we must define a pod that references the volume as done in this YAML.

File [pod-sc-pvc.yaml](pod-sc-pvc.yaml)

```
kind: Pod
apiVersion: v1
metadata:
  name: pod-sio-small
spec:
  containers:
    - name: pod-sio-small-container
      image: gcr.io/google_containers/test-webserver
      volumeMounts:
      - mountPath: /test
        name: test-data
  volumes:
    - name: test-data
      persistentVolumeClaim:
        claimName: pvc-sio-small
```

Notice that the `claimName:` attribute refers to the name of the PVC defined and deployed earlier.  Next, let us deploy the file.

```
$> kubectl create -f examples/volumes/scaleio/pod-sc-pvc.yaml
```
We can now verify that the new pod is deployed OK.
```
kubectl get pod
NAME            READY     STATUS    RESTARTS   AGE
pod-0           1/1       Running   0          23m
pod-sio-small   1/1       Running   0          5s
```
You can use the ScaleIO dashboard to verify that the new volume has one attachment.  You can verify the volume information for the pod:
```
$> kubectl describe pod pod-sio-small
...
Volumes:
  test-data:
    Type:	PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:	pvc-sio-small
    ReadOnly:	false
...
```
Lastly, you can see the volume's attachment on the Kubernetes node:
```
$> lsblk
...
scinia      252:0    0    8G  0 disk /var/lib/kubelet/pods/135986c7-dcb7-11e6-9fbf-080027c990a7/volumes/kubernetes.io~scaleio/vol-0
scinib      252:16   0   16G  0 disk /var/lib/kubelet/pods/62db442e-dcba-11e6-9fbf-080027c990a7/volumes/kubernetes.io~scaleio/sio-5fc9154ddcae11e68db708002

```
<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/scaleio/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
