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

This document shows how to provision storage in Kubernetes using the libStorage volume plugin.  It demonstrates that the libStorage volume plugin supports Kubernetes PersistentVolume (PV), PersistentVolumeClaim (PVC), StorageClass, and the Dynamic Provisioner API.  While the examples in this doc uses Virtualbox as the storage provider, the Kubernetes libStorage volume plugin will work with all storage backends supported by libStorage (see [libStorage Providers](http://libstorage.readthedocs.io/en/stable/user-guide/storage-providers/#storage-providers).

### Pre-Reqisites

* REX-Ray 0.4.0 (a libStorage server runtime), http://rexray.readthedocs.io
* VirtualBox 5.0.10+
* Knowledge of running Kubernetes

#### Deployment Options

Choose one of the following deployment setups as documented on Kubernetes website:
* [Vagrant](../../../docs/devel/local-cluster/vagrant.md) - use the Kubernetes Vagrant file to start a cluster (with Virtuabox as provider)
* [Local Setup](../../../docs/devel/local-cluster/local.md) - Setup a Linux VM running inside VirtualBox.

If you want to use a different Kubernetes setup that is not mentioned here (from scratch/bare metal, a cloud provider, etc), ensure to follow setup instructions for REX-Ray found in this doc.

*Note* - `minikube`, as of this writing,  did not support PersistentVolume other than `HostPath`.  We will update this document when it supports all other PVs.

### VirtualBox Configuration

The configuration for libStorage requires a VirtualBox VM which will host REX-Ray (the libStorage server runtime) and the Kubernetes nodes that will host pods that will consume storage.  These components (the REX-Ray server and the k8s pods) may be running on the same machine depending on your deployment setup.  The examples in this document assumes a local-cluster in one guest VM where all components are running on the same machine.

To ensure we can add new volumes to the VM, we need to adjust the storage controller's port count settings.  In VirtualBox `Settings > Storage > Controller` do the followings:

* Take note of the name attribute of the `SATA` controller.  Sometimes the name is just `SATA` and sometimes it's `SATA controller`.  We will use this name when configuring libStorage.
* Increase the number of `Port Count` for the SATA controller to 30.  This is to prevent errors when creating new volumes.

Next, configure the VirtualBox API service. The instructions assume VirtualBox running on an OSX host, however, they are similar for Linux  (consult VirtualBox documentation).  Ensure that the VirtualBox binaries are in your PATH environment variable.

First, disable the webservice authentication:

```
$> VBoxManage setproperty websrvauthlibrary null
```

Next, start the `vboxwebsrv`  VirtualBox API server.

```
$> /Applications/VirtualBox.app/Contents/MacOS/vboxwebsrv -H 0.0.0.0 -v
```

Or, if running on Linux, you will need to start the VirtualBox service:

```
$> sudo /etc/init.d/vboxweb-service start 
```

This will start the VirtualBox API server on default port `18083`.

### REX-Ray Setup

As mentioned earlier, this example uses the REX-Ray (http://rexray.readthedocs.io) which is a libStorage server runtime capable of handling storage orchestration.  It is a single binary that gets installed on a machine reachable by the Kubernetes nodes.  A suggested location for installing REX-Ray is a machine running master components (i.e. controller, api server, etc).  Keep in mind, if you are running a single node local cluster, as in this example, all of these components run on the same machine.

* SSH into the  machine to that will run the REX-Ray binary
* From the command line, install the latest stable REX-Ray binary as root:

```
# curl -sSL https://dl.bintray.com/emccode/rexray/install | sh -s -- stable
```

Once REX-Ray is installed successfully, you can validate the installation as follows:

```
$> rexray version
REX-Ray
-------
Binary: /usr/bin/rexray
SemVer: 0.6.0
OsArch: Linux-x86_64
Branch: v0.6.0
Commit: 429cfdf7272cebc5911b2865128ad37a6da5a203
Formed: Thu, 20 Oct 2016 22:06:45 EDT

libStorage
----------
SemVer: 0.3.1
OsArch: Linux-x86_64
Branch: v0.6.0
Commit: a3a561a1b94bb8b7efb4ae998540c457deb39692
Formed: Thu, 20 Oct 2016 22:05:09 EDT
```

While in an SSH session, on the machine where REX-Ray got installed, as root create director `/etc/rexray`.  Then, create and save the REX-Ray configuration file for VirtualBox as shown in the following.

File: [config.yml](config.yml)

```
#> sudo mkdir -p /etc/rexray
#> sudo tee -a /etc/rexray/config.yml << EOF
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
  volumePath: <VirtualBox-Volume-Path>
  controllerName: SATA
EOF
```

Where:

  * `virtualbox.endpoint` - the address of the VirtualBox API server
  * `virtualbox.volumePath` - the path where VirtualBox stores its virtual volume files on the host machine.  For insance, if you are running VirtualBox on a Mac, it may be something like `/Users/<your-user-id>/VirtualBox Volumes`
  * `virtualbox.controllerName` - is the SATA controller name as it is setup for the VirtualBox VM.  Ensure that it matches the name of the SATA controller configured earlier.

Now, let us further validate REX-Ray's installation by getting a list of VirtualBox volumes available:

```
$> sudo rexray volume ls
```

If everything works, you should see an output of all VirtualBox volumes on the machine similar to the followings:

```
$> sudo rexray volume ls
ID                                    Name                 Status       Size
4bdc0f31-07d6-4715-8e50-85bfa4bd2f4a  HD1.vdi              unavailable  50
030806d6-67da-42c7-987b-5211627f5d63  Haiku-Os.vdi         unavailable  10
a63c7bf6-2231-40ab-929d-490e127326d9  NewVirtualDisk1.vdi  available    41
75dca810-56ea-4c09-8284-d6cebe8a8509  Work-Ubuntu.vdi      attached     80
```

This should be all you need to run the examples.  However, if you get stuck and need to troubleshoot any of the steps for setting up REX-Ray, see the [Configuration Guide](http://rexray.readthedocs.io/en/stable/user-guide/config/).

### Starting Kubernetes

Now that all pre-requisites are out of the way, it is time to focus on Kubernets.  First, ensure that you start the Kubernertes cluster as prescribed for your setup (local-cluster, vagrant, from scratch, etc).  For this example, we used a local cluster, so we will start Kubernetes with the following command:

```
$> LOG_LEVEL=99 hack/local-up-cluster.sh
```

The local cluster script will start all necessary Kubernetes components (make sure to follow the instructions from the startup script).  Now, validate that the libStorage Kubernetes Plugin started OK.  For instance, if you are running local setup, `grep` the log files for `libstorage`:

```
$> cat /tmp/kube-controller-manager.log | grep libstorage
I0916 13:25:40.469430   13339 plugins.go:352] Loaded volume plugin "kubernetes.io/libstorage"
I0916 13:25:40.478110   13339 plugins.go:352] Loaded volume plugin "kubernetes.io/libstorage"
```

## PersistentVolumes and PersistentVolumeClaims

The following examples show how libStorage volume plugin for Kubernetes will automatically attach, format, mount, and bind-mount volumes and volume claims deployed on the cluster.  Regular PVs snd PVCs require that the volumes that they use be already created ahead of time.

As root, use REX-Ray binary to create the new volume.  Make note of the name of the volume as it will be used later.

```
$> sudo rexray volume create --size=1 vol-0001
ID                                    Name      Status     Size
0a0de085-f813-4430-8376-854432841e02  vol-0001  available  1
```

### Pods with embedded PVs

Let's create the following pod which includes a persistent volume embedded directly in the pod's spec as shown below.  Deploying the pod will attach the volume to the node and bind-mount it into the container.  Notice it uses a volume name that matches the name of the volume that was created above.

```
$> cluster/kubectl.sh create -f examples/volumes/libstorage/pod.yml
```

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
      host: http://127.0.0.1:7979
      volumeName: vol-0001
      service: virtualbox
```

Notice that the `volumes.libStorage` section of the YAML includes configuration information such as `host` and `service` that are used to communicate with the REX-Ray libStorage server runtime.

#### Validate Pod Deployment

`$> ./cluster/kubectl.sh describe pod` should return (among a list of other information) the following:

```
Volumes:
  vol-0001:
    Type:	LibStorage (a volume managed by LibStorage service)
    Host:	http://127.0.0.1:7979
    Service:	virtualbox
    VolumeName:	vol-0001
    FSType:
    ReadOnly:	false
```

#### Validate Volume Attachment

Use REX-Ray to list the volume information and notice the attachment information (compared to above).  Notice the status of the volume is updated to `attached`.

```
$> sudo rexray volume get 97ef709f-c057-463e-8212-a262cf1fa2f2
ID                                    Name      Status    Size
97ef709f-c057-463e-8212-a262cf1fa2f2  vol-0001  attached  1
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

This example shows how the Kubernetes libStorage volume plugin supports the use of PVCs to add storage to kubernetes nodes.  The first step, again, is to create the volume (if it has not been created).

#### Create the Volume

```
$> sudo rexray volume create --size=1 vol-0001
ID                                    Name      Status     Size
0a0de085-f813-4430-8376-854432841e02  vol-0001  available  1
`
```

#### Deploy PersistentVolume

Next, create/deploy a PV configuration.  Note the `spec.libStorage` configuration section in the PVC which include the `volumeName`, `host`, and `service` entries.

```
$> cluster/kubectl.sh create -f examples/volumes/libstorage/pv.yml
```

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
    host: http://127.0.0.1:7979
    service: virtualbox
    volumeName: vol-0001
```

We can validate the deployment with `$> ./cluster/kubectl.sh describe pv`. with the output below.  Notice the PV is not bound to a claim yet.

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

```
$> cluster/kubectl.sh create -f examples/volumes/libstorage/pvc.yml
```

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

We can verify the deployment of the PVC with `$> ./cluster/kubectl.sh describe pvc`.  Notice that now the claim is now bound to the PV above .

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

Next, let us deploy a pod that uses the PVC that was setup above.  Notice that the volume spec points to the PVC declared above by name `claimName: pvc-0001`.

```
$> cluster/kubectl.sh create -f examples/volumes/libstorage/pod-pvc.yml
```

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

```
$> cluster/kubectl.sh create -f examples/volumes/libstorage/sc.yml
```

File: [sc.yml](sc.yml)

```
apiVersion: storage.k8s.io/v1beta1
kind: StorageClass
metadata:
  name: sc-0001
provisioner: kubernetes.io/libstorage
parameters:
  host: "http://127.0.0.1:7979"
  service: "virtualbox"
```

Note the use of the `provisioner:` entry which specifies `kuberneties.io/libstorage`.  This tells Kubernetes which plugin to use to provision the volume for this storage class.  The storage class also uses the `parameters:` section to specify the libStorage configuration information.

### Deploy a PVC

Next, let us defined/deploy a PVC that will make use of the storage class.  Note the `annotations:` entry which specifies annotation `volume.beta.kubernetes.io/storage-class: sc-0001` which references the name of the storage class defined earlier.

```
$> cluster/kubectl.sh create -f examples/volumes/libstorage/sc-pvc.yml
```

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

```
$> cluster/kubectl.sh create -f examples/volumes/libstorage/pod-sc-pvc.yml
```

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
