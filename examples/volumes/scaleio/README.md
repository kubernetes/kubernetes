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
* A ScaleIO storage cluster with an API gateway
* SDC binary installed on Kubernetes nodes

## Kubernetes Setup Options

To run this example, you should deploy Kuberenetes using one of the followings cluster setup:
* Local cluster
* VMs (on premise or via a cloud servers)
* Baremetal (from scratch)
For instructions on deploying Kubernetes, see [this page](http://kubernetes.io/docs/getting-started-guides/).
Whichever option you have selected, ensure that you have plenty of permissions to copy built Kuberneties binaries (and ScaleIO binaries) 

## ScaleIO Setup

This document assumes you are familiar with ScaleIO and have a cluster ready.  If you are *not familiar* with ScaleIO and would like to get started, use the following resources:

  * Learn how to setup a 3-node [ScaleIO cluster on Vagrant](https://github.com/codedellemc/labs/tree/master/setup-scaleio-vagrant)
  * General instructions on [setting up ScaleIO](https://www.emc.com/products-solutions/trial-software-download/scaleio.htm)

For this demonstration, ensure that the ScaleIO `SDC` component is installed on all Kubernetes nodes where deployed pods will consume ScaleIO-backed volumes.

## Deploy Kubernetes Secret for ScaleIO

The ScaleIO plugin uses Kubernetes Secret object to store the `username` and `password` that is used to connect to the ScaleIO's gateway API server.  In this step, let us create a secret object to save the data.

To avoid storing secrets in as clear text, let us encode the ScaleIO credentials as `base64` using the following steps.

```
$> echo -n "siouser" | base64
c2lvdXNlcg==
$> echo -n "sc@l3I0" | base64
c2NAbDNJMA==
```

The previous will generate `base64-encoded` values for the username and password.  Remember to generate the credentials for your own environment (not the username/password shown above) .  Next, create a secret file, with the encoded values from above, as shown in the following.

File: [secret.yaml](secret.yaml)

```
apiVersion: v1
kind: Secret
metadata:
  name: sio-secret
type: Opaque
data:
  username: c2lvdXNlcg==
  password: c2NAbDNJMA==
```

Notice the name of the secret specified above as `sio-secret`.  It will be referred in other YAML files.  Next, deploy the secret.

```
$ kubectl create -f ./examples/volumes/scaleio/secret.yaml
```

## Deploying Pods with Persistent Volumes

The following example shows how the ScaleIO volume plugin for Kubernetes automatically attach, format, and mount a volume for a deployed pod. Static persistent volumes require that ScaleIO volume be already created ahead of time.

### Create Volume

Static persistent volumes require that the volume, to be consumed by the pod, be already created in ScaleIO.  This means, you have to create a new volume using your ScaleIO tools, or use the name of a volume that already exists.  For this demo, we assume there's a volume named `vol-0001`.  If you want to use an existing volume, ensure its name is reflected properly in the `volumeName` attribute below.

#### A note on volume name

The name of the volume is critical and is regarded as a unique identifier.  Therefore, when explicitly naming volume to use with the Kubernetes ScaleIO plugin, ensure that the names are unique throughout the entire Kubernetes cluster.

### Deploy YAML

Create a pod YAML file that declares the volume (above) to be used.

File: [pod.yaml](pod.yaml)

```
apiVersion: v1
kind: Pod
metadata:
  name: pod-0001
spec:
  containers:
  - image: gcr.io/google_containers/test-webserver
    name: pod-0001-container
    volumeMounts:
    - mountPath: /test-pd
      name: vol-0001
  volumes:
  - name: vol-0001
    scaleIO:
      gateway: http://127.0.0.1:7979
      system: scaleio
      volumeName: vol-0001
      secretRef:
        name: sio-secret
```

Make sure to update the `gatewway` to your ScaleIO gateway endpoint.  Notice the `volumeName` attribute refers to the name of an existing volume in ScaleIO.  Also, notice that  `secretRef` references the name of the secret object deployed earlier.

Next, deploy the pod.

```
$> kubectl create -f examples/volumes/scaleio/pod.yaml
```

#### Validate Persistent Volume

It may take a few seconds before the actual volume gets attached, formatted, and mounted.  You can validate that the volume is attached in ScaleIO UI.  You can also validate that the pod has now been deployed successfully with the volume.

```
kubectl describe pod pod-0001
```

## StorageClass and Dynamic Provisioning

The Kubernetes Scaleio volume plugin also supports dynamic provisioning of volumes via storage classes. In this example, we will see how the ScaleIO volume plugin can automatically provision a new volume as described in a StorageClass.

### Deploy StorageClass

Let us defined/deploy a `StorageClass` as shown in the following YAML.

File [sc.yaml](sc.yaml)

```
kind: StorageClass
apiVersion: storage.k8s.io/v1beta1
metadata:
  name: sc-0001
provisioner: kubernetes.io/scaleio
parameters:
  gateway: http://127.0.0.1:34567
  system: scaleio
  protectionDomain: default
  secretRef: sio-secret
```

Use the `parameters:` section in the yaml above to specify ScaleIO specific configuration values.  Once again, ensure the `secretRef` matches the name of the Secret object created earlier.

Next, deploy the storage class file.

```
$> kubectl create -f examples/volumes/scaleio/sc.yaml
```

### Deploy PVC for the StorageClass

The next step is to define a `PeristentVolumeClaim` that will use the storage defined in the StorageClass.

File [sc-pvc.yaml](sc-pvc.yaml)

```
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: pvc-0002
  annotations:
      volume.beta.kubernetes.io/storage-class: sc-0001
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

Note the `annotations:` entry which specifies annotation `volume.beta.kubernetes.io/storage-class: sc-0001` which in turn references the name of the storage class defined earlier.

Next, deploy PVC file for the storage class.  This step will cause the Kubernetes ScaleIO plugin to create the volume in the storage system.  You can verify that a new volume has been created within the ScaleIO administrative GUI.

```
$> kubectl create -f examples/volumes/scaleio/sc-pvc.yaml
```

### Deploy Pod for PVC and SC

The last step is to define and deploy a pod that consumes storage from the volume created by the storage class above.

File [pod-sc-pvc.yaml](pod-sc-pvc.yaml)

```
kind: Pod
apiVersion: v1
metadata:
  name: podscpvc-0001
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

Notice that the `claimName:` attribute refers to the name of the PVC defined and deployed earlier.  Next, let us deploy the file.

```
$> kubectl create -f examples/volumes/scaleio/pod-sc-pvc.yaml
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/scaleio/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
