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

# Persistent Volumes with Kubernetes + libStorage

This document shows how to use Kubernetes, libStorage (via RexRay), and  Virtualbox as the storage provider.  It demonstrates libStorage support for PersistentVolume, PersistentVolumeClaim, StorageClass, and dynamic Provisioner.

### Pre-Reqisites

* REX-Ray 0.4.0 or above (a libStorage provider), http://rexray.readthedocs.io
* VirtualBox 5.0.10+
* Knowledge of Kubernetes

#### Deployment Options

Choose one of the following deployment setups as documented on Kubernetes website:
* [Vagrant](../../../docs/devel/local-cluster/vagrant.md) - use the Kubernetes Vagrant file to start a cluster (with Virtuabox as provider)
* [Local Setup](../../../docs/devel/local-cluster/local.md) - Setup a Linux VM running inside VirtualBox
If you want to use a different Kubernetes setup that is not covered here (from scratch/bare metal, a cloud provider, etc*), ensure that you setup REXRay with the proper configuration and proper driver for the storage provider.

*Note* - `minikube` does not support PersistentVolume (as noted on the project's repository) and therefore would not work.

### VirtualBox Config

Once your virtual machine images are setup on VirtualBox, we need to update their configuration for libStorage to work properly.  For each node (VirtualBox image) where kubelet/pod will run do the following:

* Take note of the name of the IDE controller from VirtualBox (used later to configure libStorage)
* Under VirtualBox `Settings > Storage > Controller`, increase the number of `Port Count` for the controller to more than 1 (if you are going to create many volumes, 30 is  good number).

Next, configure the VirtualBox API service (the instructions assume you are running VirtuaBox on OSX).  First, disable the webservice authentication,

```
VBoxManage setproperty websrvauthlibrary null
```

Next, start the `vboxwebsrv`  VirtualBox API server.

```
/Applications/VirtualBox.app/Contents/MacOS/vboxwebsrv -H 0.0.0.0 -v
```

### REX-Ray Setup

This example uses the REX-Ray (http://rexray.readthedocs.io) as a libStorage storage orchestrator.  It is a single binary that gets installed on a machine reachable by the Kubernetes nodes.  A suggested location for installing REXRay is a machine running master components (i.e. controller, api server, etc).

* SSH into a into the machine (if needed)
* From the command line, install the latest stable REX-Ray binary as root:

```
# curl -sSL https://dl.bintray.com/emccode/rexray/install | sh -s -- stable
```

Once REX-Ray is installed successfully, you will see a confirmation output similar to the following:

```
REX-Ray
-------
Binary: /home/akutz/go/bin/rexray
SemVer: 0.4.0-rc4+10+dirty
OsArch: Linux-x86_64
Branch: release/0.4.0-rc4
Commit: 063a0794ac19af439c3ab5a01f2e6f5a4f4f85ae
Formed: Tue, 14 Jun 2016 14:23:15 CDT

libStorage
----------
SemVer: 0.1.3
OsArch: Linux-x86_64
Branch: v0.1.3
Commit: 182a626937677a081b89651598ee2eac839308e7
Formed: Tue, 14 Jun 2016 14:21:25 CDT
```

Now, as root create director `/etc/rexray`.  Then, create and save a REX-Ray configuration file for VirtualBox as shown in the following.

File: [config.yml](config.yml)

```
# sudo mkdir -p /etc/rexray
# sudo tee -a /etc/rexray/config.yml << EOF
libstorage:
  service: virtualbox
  host: tcp://127.0.0.1:7979
  embedded: true
  server:
    services:
      virtualbox:
        driver: virtualbox

virtualbox:
  endpoint: http://10.0.2.2:18083
  tls: false
  volumePath: /Users/vladimir/VirtualBox Volumes
  controllerName: SATA
EOF
```

Where:

  * `virtualbox.endpoint` - The address of the VirtualBox API server
  * `virtualbox.volumePath` - a valid path on your local machine where VirtualBox will store volume files.
  * `virtualbox.controllerName` - is the IDE controller name as it is setup for the VirtualBox VM.  Ensure that it matches the name of the IDE controller.

For further instructions about setting up REXRay, see the [Configuration Guide](http://rexray.readthedocs.io/en/stable/user-guide/config/).

### Starting Kubernetes

Ensure that you have started the Kubernertes cluster as prescribed for your setup (local, vagrant, etc).  Next, validate that the libStorage Kubernetes Plugin OK.  For instance, if you are running local setup, `grep` the log files for `libstorage`:

```
> cat /tmp/kube-controller-manager.log | grep libstorage
I0916 13:25:40.469430   13339 plugins.go:352] Loaded volume plugin "kubernetes.io/libstorage"
I0916 13:25:40.478110   13339 plugins.go:352] Loaded volume plugin "kubernetes.io/libstorage"
```

## PersistentVolumes and PersistentVolumeClaims

The following examples show how libStorage volume plugin for Kubernetes will automatically attach, format, mount, and bind-mount volumes and volume claims deployed on the cluster.  Regular PVs snd PVCs require that the volumes that they use be already created ahead of time.

So, SSH into a node where the volume will be used.  As root, use REX-Ray binary to create the new volume.  Make note of the name of the volume as it will be used later.

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

### Pods with embedded PVs

The following pod includes a persistent volume embedded directly in the pod's spec as shown below.  Notice it uses a volume name that matches the name that was created above.

File [pod.yml](pod.yml)

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
    libStorage:
      host: tcp://:7979
      volumeName: vol-0001
      service: virtualbox
```

Notice that the `volumes.libStorage` section of the YAML includes configuration information such as `Host` and `Service` that are used to communicate to the libStorage service (hosted by REX-Ray)

#### Validate Pod Deployment

`$> ./cluster/kubectl.sh describe pod` should return (among a list of other information) the following:

```
Volumes:
  vol-0001:
    Type:	LibStorage (a volume managed by LibStorage service)
    Host:	tcp://:7979
    Service:	virtualbox
    VolumeName:	vol-0001
    FSType:
    ReadOnly:	false
```

#### Validate Volume Attachment

Use REX-Ray to list the volume information and notice the attachment information (compared to above).

```
root@vagrant-ubuntu-trusty-64:/home/vagrant# rexray volume get --volumename vol-0001
attachments:
- instanceID:
    id: da4e52c9-79e7-423f-bc8c-509c022a98e0
    driver: virtualbox
  status: /Users/vladimir/VirtualBox Volumes/vol-0001
  volumeID: c3932aee-ee52-44fe-84a6-fcb426724ca5
name: vol-0001
size: 1
status: /Users/vladimir/VirtualBox Volumes/vol-0001
id: c3932aee-ee52-44fe-84a6-fcb426724ca5
type: ""
```

If you grep your `kube-controller.log` file for `libStorage` you will see the log for that attachment.  For instance, the following is from a local cluster that shows libStorage attaching the volume to the host instance.

```
> cat /tmp/kube-controller-manager.log | grep libStorage
I0916 13:51:42.541397   30692 lsattacher.go:47] libStorage: attaching volume to host 127.0.0.1
I0916 13:51:43.388893   30692 lsmgr.go:136] libStorage: attaching volume vol-0001 to host instace da4e52c9-79e7-423f-bc8c-509c022a98e0
I0916 13:51:45.496218   30692 lsmgr.go:214] libStorage: volume vol-0001 attached at device path /dev/sdb
I0916 13:51:45.496242   30692 lsattacher.go:68] libStorage: successfully attached device /dev/sdb to host 127.0.0.1
```

#### Validate Volume Formatting and Mounting

Next, grep the `kubelet.log` file to see node storage activities.  For instance, the following shows the volume being attached, formatted, mounted, and bind-mounted for the container.

```
I0916 13:51:59.320037   30706 lsattacher.go:134] libStorage: attempting to mount vol-0001:/dev/sdb as /var/lib/kubelet/plugins/kubernetes.io/libstorage/mounts/virtualbox/vol-0001
I0916 13:51:59.352386   30706 lsattacher.go:182] libStorage: formatted vol-0001:/dev/sdb [,[]], mounted as /var/lib/kubelet/plugins/kubernetes.io/libstorage/mounts/virtualbox/vol-0001
I0916 13:51:59.357896   30706 lsvolume.go:128] libStorage: successfully bind-mounted vol-0001:/var/lib/kubelet/plugins/kubernetes.io/libstorage/mounts/virtualbox/vol-0001 as /var/lib/kubelet/pods/3a0457d5-7c36-11e6-ab77-08002795680c/volumes/kubernetes.io~libstorage/vol-0001
```

### Pods with PersistentVolume Claims

This example shows how the libStorage volume plugin supports the use of PVCs to add storage to kubernetes nodes.  The first step, again, is to create the volume (assume we're starting fresh).

#### Create the Volume

```
#> rexray volume create --volumename="vol-0001" --size=1
```

#### Deploy PersistentVolume

Next, create/deploy a PV configuration.  Note the `libStorage` configuration section in the PVC which include the `volumeName`, `host`, and `service` entries.

File: [pv.yml](pv.yml)

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
  libStorage:
    host: tcp://:7979
    service: virtualbox
    volumeName: vol-0001
```

We can validate the deployment with `$> ./cluster/kubectl.sh describe pv`. with the output below.  Notice it is not bound to a claim yet.

```
Name:		pv-0001
Labels:		<none>
Status:		Available
Claim:
Reclaim Policy:	Retain
Access Modes:	RWO
Capacity:	1Gi
Message:
Source:
No events.
```

#### Deploy PersistentVolumeClaim

Next, we define/deploy a `PerisistentVolumeClaim`  as shown below.

File: [pvc.yml](pvc.yml)

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

We can verify the deployment of the PVC with `$> ./cluster/kubectl.sh describe pvc`.  Notice that the claim is now bound to the PV above .

```
Name:		pvc-0001
Namespace:	default
Status:		Bound
Volume:		pv-0001
Labels:		<none>
Capacity:	1Gi
Access Modes:	RWO
No events.
```

#### Deploy Pod with PVC

Next, let us deploy a pod that uses the PVC setup above.  Notice that the volume spec points to the PVC declared above by name.

File: [pod-pvc.yml](pod-pvc.yml)

```
kind: Pod
apiVersion: v1
metadata:
  name: podpvc-0001
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

Now, we can validate the claim with a `kubectl describe pod` which will show (among other information) the following.

```
Volumes:
  test-data:
    Type:	PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:	pvc-0001
    ReadOnly:	false
```

Once again, when we grep the log files, we see the libStorage volume plugin attaching the volume.

```
 cat /tmp/kube-controller-manager.log | grep libStorage
I0916 15:20:00.338945   28740 lsattacher.go:47] libStorage: attaching volume to host 127.0.0.1
I0916 15:20:01.199797   28740 lsmgr.go:136] libStorage: attaching volume vol-0001 to host instace da4e52c9-79e7-423f-bc8c-509c022a98e0
I0916 15:20:03.332405   28740 lsmgr.go:214] libStorage: volume vol-0001 attached at device path /dev/sdd
I0916 15:20:03.332431   28740 lsattacher.go:68] libStorage: successfully attached device /dev/sdd to host 127.0.0.1
```

And here we see the volume being mounted by the libStorage volume plugin.

```
I0916 15:20:05.096002   28744 lsvolume.go:98] libStorage: bind-mounting vol-0001:/var/lib/kubelet/plugins/kubernetes.io/libstorage/mounts/virtualbox/vol-0001 to /var/lib/kubelet/pods/8fb9b742-7c42-11e6-bbee-08002795680c/volumes/kubernetes.io~libstorage/vol-0001
I0916 15:20:05.102746   28744 lsvolume.go:128] libStorage: successfully bind-mounted vol-0001:/var/lib/kubelet/plugins/kubernetes.io/libstorage/mounts/virtualbox/vol-0001 as /var/lib/kubelet/pods/8fb9b742-7c42-11e6-bbee-08002795680c/volumes/kubernetes.io~libstorage/vol-0001
```

## StorageClass and Dynamic Provisioning

The Kubernetes libStorage volume plugin also supports dynamic provisioning of volumes via storage classes.  In this example, we will see how the libStorage volume plugin can create a new volume as described in a `StorageClass`.

### Deploy a StorageClass

First, let's defined/deploy a `StorageClass` as shown in the following YAML.

File: [sc.yml](sc.yml)

```
apiVersion: storage.k8s.io/v1beta1
kind: StorageClass
metadata:
  name: sc-0001
provisioner: kubernetes.io/libstorage
parameters:
  host: "tcp://:7979"
  service: "virtualbox"
```

Note the use of the `provisioner:` entry which specifies `kuberneties.io/libstorage`.  This tells Kubernetes which plugin to use to provision the volume for this storage class.  The storage class also uses the `parameters:` section to specify the libStorage configuration information.

### Deploy a PVC

Next, let us defined/deploy a PVC that will make use of the storage class.  Note the `annotations:` entry which specifies annotation `volume.beta.kubernetes.io/storage-class: sc-0001` which references the name of the storage class defined earlier.

File: [sc-pvc.yml](sc-pvc.yml)

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

#### Validate deployment

When the claim is deployed, it  triggers the creation of a new volume.  This can be validated by running `$> ./cluster/kubectl.sh describe pvc` which shows the created volume.

```
Name:		pvc-0002
Namespace:	default
Status:		Bound
Volume:		kubernetes-dynamic-pvc-8e000001-7c46-11e6-abe3-08002795680c
Labels:		<none>
Capacity:	1Gi
Access Modes:	RWO
```

We can further validate the new volume using REX-Ray itself using `#> rexray volume get --volumename kubernetes-dynamic-pvc-8e000001-7c46-11e6-abe3-08002795680c` which will print something similar to the following.

```
name: kubernetes-dynamic-pvc-8e000001-7c46-11e6-abe3-08002795680c
size: 1
status: /Users/user/VirtualBox Volumes/kubernetes-dynamic-pvc-8e000001-7c46-11e6-abe3-08002795680c
id: cbbeccf7-49a4-490a-8790-8631e746ac0d
```

Lastly we can validate the volume being created by looking at the Kubernetes log.

```
I0916 15:48:35.334619    4279 lsprovisioner.go:31] libStorage: attempting to provision volume %s
I0916 15:48:36.205089    4279 lsmgr.go:97] libStorage: provisioning volume kubernetes-dynamic-pvc-8e000001-7c46-11e6-abe3-08002795680c
I0916 15:48:36.371489    4279 lsmgr.go:116] libStorage: successfully provisioned volume kubernetes-dynamic-pvc-8e000001-7c46-11e6-abe3-08002795680c
```

### Deploy a Pod

Lastly, we can deploy a pod that makes use of the PVC which uses the dynamically provisioned storage class as is done with the following pod definition.

File: [pod-sc-pvc.yml](pod-sc-pvc.yml)

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

#### Validation

Note that the pod is using the claim that was created earlier with the storage class annotation.  When this pod is deployed, it will attach, format, mount and bind-mount the volume that was dynamically created earlier.  When we grep kube-controller.log we can see the volume being attached to the host.

```
I0916 16:03:48.283186   21061 lsmgr.go:214] libStorage: volume kubernetes-dynamic-pvc-a11a980c-7c48-11e6-9bd2-08002795680c attached at device path /dev/sde
I0916 16:03:48.283211   21061 lsattacher.go:68] libStorage: successfully attached device /dev/sde to host 127.0.0.1
```

Next, we can see the libStorage plugin formatting, mounting and bind mounting the volume.

```
I0916 16:03:50.053742   21062 lsattacher.go:182] libStorage: formatted kubernetes-dynamic-pvc-a11a980c-7c48-11e6-9bd2-08002795680c:/dev/sde [,[]], mounted as /var/lib/kubelet/plugins/kubernetes.io/libstorage/mounts/virtualbox/kubernetes-dynamic-pvc-a11a980c-7c48-11e6-9bd2-08002795680c
I0916 16:03:50.053742   21062 lsattacher.go:182] libStorage: formatted kubernetes-dynamic-pvc-a11a980c-7c48-11e6-9bd2-08002795680c:/dev/sde [,[]], mounted as /var/lib/kubelet/plugins/kubernetes.io/libstorage/mounts/virtualbox/kubernetes-dynamic-pvc-a11a980c-7c48-11e6-9bd2-08002795680c
I0916 16:03:50.057861   21062 lsvolume.go:128] libStorage: successfully bind-mounted kubernetes-dynamic-pvc-a11a980c-7c48-11e6-9bd2-08002795680c:/var/lib/kubelet/plugins/kubernetes.io/libstorage/mounts/virtualbox/kubernetes-dynamic-pvc-a11a980c-7c48-11e6-9bd2-08002795680c as /var/lib/kubelet/pods/ac4a2d8f-7c48-11e6-9bd2-08002795680c/volumes/kubernetes.io~libstorage/kubernetes-dynamic-pvc-a11a980c-7c48-11e6-9bd2-08002795680c
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/libstorage/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
