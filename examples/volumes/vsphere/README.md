# vSphere Volume

  - [Prerequisites](#prerequisites)
  - [Examples](#examples)
    - [Volumes](#volumes)
    - [Persistent Volumes](#persistent-volumes)
    - [Storage Class](#storage-class)
    - [Virtual SAN policy support inside Kubernetes](#virtual-san-policy-support-inside-kubernetes)
    - [Stateful Set](#stateful-set)

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

### Virtual SAN policy support inside Kubernetes

  Vsphere Infrastructure(VI) Admins will have the ability to specify custom Virtual SAN Storage Capabilities during dynamic volume provisioning. You can now define storage requirements, such as performance and availability, in the form of storage capabilities during dynamic volume provisioning. The storage capability requirements are converted into a Virtual SAN policy which are then pushed down to the Virtual SAN layer when a persistent volume (virtual disk) is being created. The virtual disk is distributed across the Virtual SAN datastore to meet the requirements.

  The official [VSAN policy documentation](https://pubs.vmware.com/vsphere-65/index.jsp?topic=%2Fcom.vmware.vsphere.virtualsan.doc%2FGUID-08911FD3-2462-4C1C-AE81-0D4DBC8F7990.html) describes in detail about each of the individual storage capabilities that are supported by VSAN. The user can specify these storage capabilities as part of storage class defintion based on his application needs.

  The policy settings can be one or more of the following:

  * *hostFailuresToTolerate*: represents NumberOfFailuresToTolerate
  * *diskStripes*: represents NumberofDiskStripesPerObject
  * *objectSpaceReservation*: represents ObjectSpaceReservation
  * *cacheReservation*: represents FlashReadCacheReservation
  * *iopsLimit*: represents IOPSLimitForObject
  * *forceProvisioning*: represents if volume must be Force Provisioned

  __Note: Here you don't need to create persistent volume it is created for you.__
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
          hostFailuresToTolerate: "2"
          cachereservation: "20"
      ```
      [Download example](vsphere-volume-sc-vsancapabilities.yaml?raw=true)

      Here a persistent volume will be created with the Virtual SAN capabilities - hostFailuresToTolerate to 2 and cachereservation is 20% read cache reserved for storage object. Also the persistent volume will be *zeroedthick* disk.
      The official [VSAN policy documentation](https://pubs.vmware.com/vsphere-65/index.jsp?topic=%2Fcom.vmware.vsphere.virtualsan.doc%2FGUID-08911FD3-2462-4C1C-AE81-0D4DBC8F7990.html) describes in detail about each of the individual storage capabilities that are supported by VSAN and can be configured on the virtual disk.

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
          hostFailuresToTolerate: "2"
          cachereservation: "20"
      ```

      [Download example](vsphere-volume-sc-vsancapabilities-with-datastore.yaml?raw=true)

      __Note: If you do not apply a storage policy during dynamic provisioning on a VSAN datastore, it will use a default Virtual SAN policy.__

      Creating the storageclass:

      ``` bash
      $ kubectl create -f examples/volumes/vsphere/vsphere-volume-sc-vsancapabilities.yaml
      ```

      Verifying storage class is created:

      ``` bash
      $ kubectl describe storageclass fast
      Name:		fast
      Annotations:	<none>
      Provisioner:	kubernetes.io/vsphere-volume
      Parameters:	diskformat=zeroedthick, hostFailuresToTolerate="2", cachereservation="20"
      No events.
      ```

  2. Create Persistent Volume Claim.

      See example:

      ```yaml
      kind: PersistentVolumeClaim
      apiVersion: v1
      metadata:
        name: pvcsc-vsan
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
      $ kubectl describe pvc pvcsc-vsan
      Name:		pvcsc-vsan
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
      Claim:		default/pvcsc-vsan
      Reclaim Policy:	Delete
      Access Modes:	RWO
      Capacity:	2Gi
      Message:
      Source:
          Type:	vSphereVolume (a Persistent Disk resource in vSphere)
          VolumePath:	[VSANDatastore] kubevols/kubernetes-dynamic-pvc-80f7b5c1-94b6-11e6-a24f-005056a79d2d.vmdk
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
            mountPath: /test
        volumes:
        - name: test-volume
          persistentVolumeClaim:
            claimName: pvcsc-vsan
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

###  Stateful Set

vSphere volumes can be consumed by Stateful Sets.

  1. Create a storage class that will be used by the ```volumeClaimTemplates``` of a Stateful Set.

      See example:

      ```yaml
      kind: StorageClass
      apiVersion: storage.k8s.io/v1beta1
      metadata:
        name: thin-disk
      provisioner: kubernetes.io/vsphere-volume
      parameters:
          diskformat: thin
      ```

      [Download example](simple-storageclass.yaml)

  2. Create a Stateful set that consumes storage from the Storage Class created.

     See example:
     ```yaml
     ---
     apiVersion: v1
     kind: Service
     metadata:
       name: nginx
       labels:
         app: nginx
     spec:
       ports:
       - port: 80
         name: web
       clusterIP: None
       selector:
         app: nginx
     ---
     apiVersion: apps/v1beta1
     kind: StatefulSet
     metadata:
       name: web
     spec:
       serviceName: "nginx"
       replicas: 14
       template:
         metadata:
           labels:
             app: nginx
         spec:
           containers:
           - name: nginx
             image: gcr.io/google_containers/nginx-slim:0.8
             ports:
             - containerPort: 80
               name: web
             volumeMounts:
             - name: www
               mountPath: /usr/share/nginx/html
       volumeClaimTemplates:
       - metadata:
           name: www
           annotations:
             volume.beta.kubernetes.io/storage-class: thin-disk
         spec:
           accessModes: [ "ReadWriteOnce" ]
           resources:
             requests:
               storage: 1Gi
     ```
     This will create Persistent Volume Claims for each replica and provision a volume for each claim if an existing volume could be bound to the claim.

     [Download example](simple-statefulset.yaml)

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/vsphere/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
