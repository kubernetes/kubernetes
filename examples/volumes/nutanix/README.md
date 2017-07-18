### Persistent Volumes
  1. Use pre-existing volume created on nutanix cluster.
     Find out volume uuid and iscsi target by using 'acli vm.get <volume-name>' on nutanix cluster.

  2. Create pod using this volume.

     Example:

     ```yaml
        apiVersion: v1
        kind: Pod
        metadata:
          name: ntnxpd
        spec:
          containers:
          - name: ntnxpd-rw
            image: nginx
            volumeMounts:
            - mountPath: /var/lib/www/html
              name: ntnxpd-rw
          volumes:
          - name: ntnxpd-rw
            nutanixVolume:
              user: admin
              password: Nutanix.123
              prismEndPoint: 10.4.65.155:9440
              dataServiceEndPoint: 10.4.65.156:3260 
              volumeUUID: 4777ca23-5807-4e4d-9a28-e0378f58d31a
              iscsiTarget: iqn.2010-06.com.nutanix:nutanix-k8s-volume-4777ca23-5807-4e4d-9a28-e0378f58d31a
              volumeName: nutanix-k8s-volume
              fsType: ext4
              readOnly: false
     ```

     Creating the pod using kubectl command.

     ```bash
     kubectl create -f example/volumes/nutanix/pod-pv.yaml
     ```

### Dynamic Persistent Volumes

   You do not need to provision volume in nutanix cluster. Persistent volume is dynamically created.

   Example 1:
   
   Create a replication controller with nutanix dynamic volume provisioning.
   
   1. Create Storage Class
   
   Storage Class could be silver for hybrid storage and gold for flash based storage.
   
   ```yaml
       kind: StorageClass
       apiVersion: storage.k8s.io/v1
       metadata:
         name: silver
       provisioner: kubernetes.io/nutanix-volume
       parameters:
         prismEndPoint: 10.4.65.155:9440
         dataServiceEndPoint: 10.4.65.156:3260
         user: admin
         password: Nutanix.123
         storageContainer: default-container
         fsType: xfs
   ```
    
   ```bash
     kubectl create -f example/volumes/nutanix/silver.yaml
   ```
   2. Create PVC

   ```yaml
       kind: PersistentVolumeClaim
       apiVersion: v1
       metadata:
         name: claim1
       spec:
         accessModes:
         - ReadWriteOnce
         resources:
           requests:
             storage: 3Gi
         storageClassName: silver
   ```

   ```bash
       kubectl create -f example/volumes/nutanix/pvc-silver.yaml
   ```
   3. Create RC

   ```yaml
       apiVersion: v1
       kind: ReplicationController
       metadata:
         name: server
       spec:
         replicas: 1
         selector:
           role: server
         template:
           metadata:
             labels:
               role: server
           spec:
             containers:
             - name: server
               image: nginx
               volumeMounts:
               - mountPath: /var/lib/www/html
                 name: mypvc
             volumes:
             - name: mypvc
               persistentVolumeClaim:
                 claimName: claim1
   ```

   ```bash
       kubectl create -f example/volumes/nutanix/rc-silver.yaml
   ```
  
  Example 2:
   
  Use secret and create replication controller with nutanix dynamic volume provisioning.

  1. Create a secret for accessing nutanix cluster
    ```yaml
        apiVersion: v1
        kind: Secret
        metadata:
          name: ntnx-secret
          namespace: default
        data:
          # base64 encoded user:password. E.g.: echo -n "admin:Nutanix.123" | base64
          key: YWRtaW46TnV0YW5peC4xMjM=
        type: kubernetes.io/nutanix-volume
    ```
 
  2. Create Storage Class

    ```yaml
        kind: StorageClass
        apiVersion: storage.k8s.io/v1
        metadata:
          name: silver
        provisioner: kubernetes.io/nutanix-volume
        parameters:
          prismEndPoint: 10.4.65.155:9440
          dataServiceEndPoint: 10.4.65.156:3260
          secretName: ntnx-secret
          password: Nutanix.123
          storageContainer: default-container
          fsType: xfs
    ```
    
    ```bash
        kubectl create -f example/volumes/nutanix/silver-secret.yaml
    ```

   3. Create PVC
   
    See example 1 above.
   
   4. Create RC

    See example 1 above. 