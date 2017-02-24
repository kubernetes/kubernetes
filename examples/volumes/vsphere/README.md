# vSphere Volume

  - [Prerequisites](#prerequisites)
  - [Examples](#examples)
    - [Volumes](#volumes)
    - [Persistent Volumes](#persistent-volumes)
    - [Storage Class](#storage-class)

## Prerequisites

- Kubernetes with vSphere Cloud Provider configured.
  For cloudprovider configuration please refer [vSphere getting started guide](http://kubernetes.io/docs/getting-started-guides/vsphere/).

## Examples

### Volumes

  1. Create VMDK.

      First ssh into ESX and then use following command to create vmdk,

      ```shell
      vmkfstools -c 2G /vmfs/volumes/datastore1/volumes/myDisk.vmdk
      ```

  2. Create Pod which uses 'myDisk.vmdk'.

     See example

     ```yaml
        apiVersion: v1
        kind: Pod
        metadata:
          name: test-vmdk
        spec:
          containers:
          - image: gcr.io/google_containers/test-webserver
            name: test-container
            volumeMounts:
            - mountPath: /test-vmdk
              name: test-volume
          volumes:
          - name: test-volume
            # This VMDK volume must already exist.
            vsphereVolume:
              volumePath: "[datastore1] volumes/myDisk"
              fsType: ext4
     ```

     [Download example](vsphere-volume-pod.yaml?raw=true)

     Creating the pod:

     ``` bash
     $ kubectl create -f examples/volumes/vsphere/vsphere-volume-pod.yaml
     ```

     Verify that pod is running:

     ```bash
     $ kubectl get pods test-vmdk
     NAME      READY     STATUS    RESTARTS   AGE
     test-vmdk   1/1     Running   0          48m
     ```

### Persistent Volumes

  1. Create VMDK.

      First ssh into ESX and then use following command to create vmdk,

      ```shell
      vmkfstools -c 2G /vmfs/volumes/datastore1/volumes/myDisk.vmdk
      ```

  2. Create Persistent Volume.

      See example:

      ```yaml
      apiVersion: v1
      kind: PersistentVolume
      metadata:
        name: pv0001
      spec:
        capacity:
          storage: 2Gi
        accessModes:
          - ReadWriteOnce
        persistentVolumeReclaimPolicy: Retain
        vsphereVolume:
          volumePath: "[datastore1] volumes/myDisk"
          fsType: ext4
      ```

      [Download example](vsphere-volume-pv.yaml?raw=true)

      Creating the persistent volume:

      ``` bash
      $ kubectl create -f examples/volumes/vsphere/vsphere-volume-pv.yaml
      ```

      Verifying persistent volume is created:

      ``` bash
      $ kubectl describe pv pv0001
      Name:		pv0001
      Labels:		<none>
      Status:		Available
      Claim:
      Reclaim Policy:	Retain
      Access Modes:	RWO
      Capacity:	2Gi
      Message:
      Source:
          Type:	vSphereVolume (a Persistent Disk resource in vSphere)
          VolumePath:	[datastore1] volumes/myDisk
          FSType:	ext4
      No events.
      ```

  3. Create Persistent Volume Claim.

      See example:

      ```yaml
      kind: PersistentVolumeClaim
      apiVersion: v1
      metadata:
        name: pvc0001
      spec:
        accessModes:
          - ReadWriteOnce
        resources:
          requests:
            storage: 2Gi
      ```

      [Download example](vsphere-volume-pvc.yaml?raw=true)

      Creating the persistent volume claim:

      ``` bash
      $ kubectl create -f examples/volumes/vsphere/vsphere-volume-pvc.yaml
      ```

      Verifying persistent volume claim is created:

      ``` bash
      $ kubectl describe pvc pvc0001
      Name:		pvc0001
      Namespace:	default
      Status:		Bound
      Volume:		pv0001
      Labels:		<none>
      Capacity:	2Gi
      Access Modes:	RWO
      No events.
      ```

  3. Create Pod which uses Persistent Volume Claim.

      See example:

      ```yaml
      apiVersion: v1
      kind: Pod
      metadata:
        name: pvpod
      spec:
        containers:
        - name: test-container
          image: gcr.io/google_containers/test-webserver
          volumeMounts:
          - name: test-volume
            mountPath: /test-vmdk
        volumes:
        - name: test-volume
          persistentVolumeClaim:
            claimName: pvc0001
      ```

      [Download example](vsphere-volume-pvcpod.yaml?raw=true)

      Creating the pod:

      ``` bash
      $ kubectl create -f examples/volumes/vsphere/vsphere-volume-pvcpod.yaml
      ```

      Verifying pod is created:

      ``` bash
      $ kubectl get pod pvpod
      NAME      READY     STATUS    RESTARTS   AGE
      pvpod       1/1     Running   0          48m        
      ```

### Storage Class

  __Note: Here you don't need to create vmdk it is created for you.__
  1. Create Storage Class.

      Example 1:

      ```yaml
      kind: StorageClass
      apiVersion: storage.k8s.io/v1beta1
      metadata:
        name: fast
      provisioner: kubernetes.io/vsphere-volume
      parameters:
          diskformat: zeroedthick
      ```

      [Download example](vsphere-volume-sc-fast.yaml?raw=true)

      You can also specify the datastore in the Storageclass as shown in example 2. The volume will be created on the datastore specified in the storage class.
      This field is optional. If not specified as shown in example 1, the volume will be created on the datastore specified in the vsphere config file used to initialize the vSphere Cloud Provider.

      Example 2:
 
      ```yaml
      kind: StorageClass
      apiVersion: storage.k8s.io/v1beta1
      metadata:
        name: fast
      provisioner: kubernetes.io/vsphere-volume
      parameters:
          diskformat: zeroedthick
          datastore: VSANDatastore
      ```

      [Download example](vsphere-volume-sc-with-datastore.yaml?raw=true)
      Creating the storageclass:

      ``` bash
      $ kubectl create -f examples/volumes/vsphere/vsphere-volume-sc-fast.yaml
      ```

      Verifying storage class is created:

      ``` bash
      $ kubectl describe storageclass fast 
      Name:		fast
      Annotations:	<none>
      Provisioner:	kubernetes.io/vsphere-volume
      Parameters:	diskformat=zeroedthick
      No events.        
      ```

  2. Create Persistent Volume Claim.

      See example:

      ```yaml
      kind: PersistentVolumeClaim
      apiVersion: v1
      metadata:
        name: pvcsc001
        annotations:
          volume.beta.kubernetes.io/storage-class: fast
      spec:
        accessModes:
          - ReadWriteOnce
        resources:
          requests:
            storage: 2Gi
      ```

      [Download example](vsphere-volume-pvcsc.yaml?raw=true)

      Creating the persistent volume claim:

      ``` bash
      $ kubectl create -f examples/volumes/vsphere/vsphere-volume-pvcsc.yaml
      ```

      Verifying persistent volume claim is created:

      ``` bash
      $ kubectl describe pvc pvcsc001
      Name:		pvcsc001
      Namespace:	default
      Status:		Bound
      Volume:		pvc-80f7b5c1-94b6-11e6-a24f-005056a79d2d
      Labels:		<none>
      Capacity:	2Gi
      Access Modes:	RWO
      No events.
      ```

      Persistent Volume is automatically created and is bounded to this pvc.

      Verifying persistent volume claim is created:

      ``` bash
      $ kubectl describe pv pvc-80f7b5c1-94b6-11e6-a24f-005056a79d2d
      Name:		pvc-80f7b5c1-94b6-11e6-a24f-005056a79d2d
      Labels:		<none>
      Status:		Bound
      Claim:		default/pvcsc001
      Reclaim Policy:	Delete
      Access Modes:	RWO
      Capacity:	2Gi
      Message:
      Source:
          Type:	vSphereVolume (a Persistent Disk resource in vSphere)
          VolumePath:	[datastore1] kubevols/kubernetes-dynamic-pvc-80f7b5c1-94b6-11e6-a24f-005056a79d2d.vmdk
          FSType:	ext4
      No events.
      ```

      __Note: VMDK is created inside ```kubevols``` folder in datastore which is mentioned in 'vsphere' cloudprovider configuration.
      The cloudprovider config is created during setup of Kubernetes cluster on vSphere.__

  3. Create Pod which uses Persistent Volume Claim with storage class.

      See example:

      ```yaml
      apiVersion: v1
      kind: Pod
      metadata:
        name: pvpod
      spec:
        containers:
        - name: test-container
          image: gcr.io/google_containers/test-webserver
          volumeMounts:
          - name: test-volume
            mountPath: /test-vmdk
        volumes:
        - name: test-volume
          persistentVolumeClaim:
            claimName: pvcsc001
      ```

      [Download example](vsphere-volume-pvcscpod.yaml?raw=true)

      Creating the pod:

      ``` bash
      $ kubectl create -f examples/volumes/vsphere/vsphere-volume-pvcscpod.yaml
      ```

      Verifying pod is created:

      ``` bash
      $ kubectl get pod pvpod
      NAME      READY     STATUS    RESTARTS   AGE
      pvpod       1/1     Running   0          48m        
      ```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/vsphere/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
