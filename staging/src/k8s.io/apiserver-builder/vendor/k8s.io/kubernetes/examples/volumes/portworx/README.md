# Portworx Volume

  - [Portworx](#portworx)
  - [Prerequisites](#prerequisites)
  - [Examples](#examples)
    - [Using Pre-provisioned Portworx Volumes](#pre-provisioned)
      - [Running Pod](#running-pod)
      - [Persistent Volumes](#persistent-volumes)
    - [Using Dynamic Provisioning](#dynamic-provisioning)
      - [Storage Class](#storage-class)

## Portworx

[Portworx](http://www.portworx.com) can be used as a storage provider for your Kubernetes cluster. Portworx pools your servers capacity and turns your servers
or cloud instances into converged, highly available compute and storage nodes

## Prerequisites

- A Portworx instance running on all of your Kubernetes nodes. For
  more information on how you can install Portworx can be found [here](http://docs.portworx.com)

## Examples

The following examples assumes that you already have a running Kubernetes cluster with Portworx installed on all nodes.

### Using Pre-provisioned Portworx Volumes

  Create a Volume using Portworx CLI.
  On one of the Kubernetes nodes with Portworx installed run the following command

  ```shell
  /opt/pwx/bin/pxctl volume create <vol-id> --size <size> --fs <fs-type>
  ```

#### Running Pods

   Create Pod which uses Portworx Volumes

   Example spec:

   ```yaml
   apiVersion: v1
   kind: Pod
   metadata:
    name: test-portworx-volume-pod
   spec:
     containers:
     - image: gcr.io/google_containers/test-webserver
       name: test-container
       volumeMounts:
       - mountPath: /test-portworx-volume
         name: test-volume
     volumes:
     - name: test-volume
       # This Portworx volume must already exist.
       portworxVolume:
         volumeID: "<vol-id>"
         fsType: "<fs-type>"
   ```

   [Download example](portworx-volume-pod.yaml?raw=true)

   Make sure to replace <vol-id> and <fs-type> in the above spec with
   the ones that you used while creating the volume.

   Create the Pod.

   ``` bash
   $ kubectl create -f examples/volumes/portworx/portworx-volume-pod.yaml
   ```

   Verify that pod is running:

   ```bash
   $ kubectl.sh get pods
     NAME                       READY     STATUS    RESTARTS   AGE
     test-portworx-volume-pod   1/1       Running   0          16s
   ```

#### Persistent Volumes

  1. Create Persistent Volume.

      Example spec:

      ```yaml
      apiVersion: v1
      kind: PersistentVolume
      metadata:
        name: <vol-id>
      spec:
        capacity:
          storage: <size>Gi
        accessModes:
          - ReadWriteOnce
        persistentVolumeReclaimPolicy: Retain
        portworxVolume:
          volumeID: "<vol-id>"
          fsType:   "<fs-type>"
      ```

      Make sure to replace <vol-id>, <size> and <fs-type> in the above spec with
      the ones that you used while creating the volume.

      [Download example](portworx-volume-pv.yaml?raw=true)

      Creating the persistent volume:

      ``` bash
      $ kubectl create -f examples/volumes/portworx/portworx-volume-pv.yaml
      ```

      Verifying persistent volume is created:

      ``` bash
      $ kubectl describe pv pv0001
      Name: 	        pv0001
      Labels:		<none>
      StorageClass:
      Status:		Available
      Claim:
      Reclaim Policy:	Retain
      Access Modes:	RWO
      Capacity:	2Gi
      Message:
      Source:
      Type:	        PortworxVolume (a Portworx Persistent Volume resource)
      VolumeID:	        pv0001
      FSType:           ext4
      No events.
      ```

  2. Create Persistent Volume Claim.

      Example spec:

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
            storage: <size>Gi
      ```

      [Download example](portworx-volume-pvc.yaml?raw=true)

      Creating the persistent volume claim:

      ``` bash
      $ kubectl create -f examples/volumes/portworx/portworx-volume-pvc.yaml
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
            mountPath: /test-portworx-volume
        volumes:
        - name: test-volume
          persistentVolumeClaim:
            claimName: pvc0001
      ```

      [Download example](portworx-volume-pvcpod.yaml?raw=true)

      Creating the pod:

      ``` bash
      $ kubectl create -f examples/volumes/portworx/portworx-volume-pvcpod.yaml
      ```

      Verifying pod is created:

      ``` bash
      $ kubectl get pod pvpod
      NAME      READY     STATUS    RESTARTS   AGE
      pvpod       1/1     Running   0          48m        
      ```

### Using Dynamic Provisioning

Using Dynamic Provisioning and Storage Classes you don't need to
create Portworx volumes out of band and they will be created automatically.

#### Storage Class

  Using Storage Classes objects an admin can define the different classes of Portworx Volumes
  that are offered in a cluster. Following are the different parameters that can be used to define a Portworx
  Storage Class

  * `fs`: filesystem to be laid out: none|xfs|ext4 (default: `ext4`)
  * `block_size`: block size in Kbytes (default: `32`)
  * `repl`: replication factor [1..3] (default: `1`)
  * `io_priority`: IO Priority: [high|medium|low] (default: `low`)
  * `snap_interval`: snapshot interval in minutes, 0 disables snaps (default: `0`)
  * `aggregation_level`: specifies the number of replication sets the volume can be aggregated from (default: `1`)
  * `ephemeral`: ephemeral storage [true|false] (default `false`)


  1. Create Storage Class.

     See example:

     ```yaml
     kind: StorageClass
     apiVersion: storage.k8s.io/v1beta1
     metadata:
       name: portworx-io-priority-high
     provisioner: kubernetes.io/portworx-volume
     parameters:
       repl: "1"
       snap_interval:   "70"
       io_priority:  "high"
     ```

     [Download example](portworx-volume-sc-high.yaml?raw=true)

     Creating the storageclass:

     ``` bash
     $ kubectl create -f examples/volumes/portworx/portworx-volume-sc-high.yaml
     ```

     Verifying storage class is created:

     ``` bash
     $ kubectl describe storageclass portworx-io-priority-high
       Name: 	        portworx-io-priority-high
       IsDefaultClass:	No
       Annotations:	<none>
       Provisioner:	kubernetes.io/portworx-volume
       Parameters:	io_priority=high,repl=1,snapshot_interval=70
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
         volume.beta.kubernetes.io/storage-class: portworx-io-priority-high
     spec:
       accessModes:
         - ReadWriteOnce
       resources:
         requests:
           storage: 2Gi
     ```

     [Download example](portworx-volume-pvcsc.yaml?raw=true)

     Creating the persistent volume claim:

     ``` bash
     $ kubectl create -f examples/volumes/portworx/portworx-volume-pvcsc.yaml
     ```

     Verifying persistent volume claim is created:

     ``` bash
     $ kubectl describe pvc pvcsc001
     Name:	      pvcsc001
     Namespace:      default
     StorageClass:   portworx-io-priority-high
     Status:	      Bound
     Volume:         pvc-e5578707-c626-11e6-baf6-08002729a32b
     Labels:	      <none>
     Capacity:	      2Gi
     Access Modes:   RWO
     No Events
     ```

     Persistent Volume is automatically created and is bounded to this pvc.

     Verifying persistent volume claim is created:

     ``` bash
     $ kubectl describe pv pvc-e5578707-c626-11e6-baf6-08002729a32b
     Name: 	      pvc-e5578707-c626-11e6-baf6-08002729a32b
     Labels:         <none>
     StorageClass:   portworx-io-priority-high
     Status:	      Bound
     Claim:	      default/pvcsc001
     Reclaim Policy: Delete
     Access Modes:   RWO
     Capacity:	      2Gi
     Message:
     Source:
         Type:	      PortworxVolume (a Portworx Persistent Volume resource)
	 VolumeID:   374093969022973811
     No events.
     ```

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
           mountPath: /test-portworx-volume
     volumes:
     - name: test-volume
       persistentVolumeClaim:
         claimName: pvcsc001
     ```

     [Download example](portworx-volume-pvcscpod.yaml?raw=true)

     Creating the pod:

     ``` bash
     $ kubectl create -f examples/volumes/portworx/portworx-volume-pvcscpod.yaml
     ```

     Verifying pod is created:

     ``` bash
     $ kubectl get pod pvpod
     NAME      READY     STATUS    RESTARTS   AGE
     pvpod       1/1     Running   0          48m        
     ```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/portworx/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
