# Kubernetes Persistent Volume Plugin For Blob and Managed Disks Samples

This repo contains samples that works with the new Azure persistent volume plugin for Kubernetes. The plugin is expected to be in v1.8 release then will become part of Azure ACS 


## What does the plugin do? 

1. Provision PVC based on Azure Managed Disks and Blob Disks
2. Perform consistent attach/detach/mount/unmount and format when needed for disks 
3. Supports both standard and premium LRS storage accounts.

## Get Started

### Using the Samples
The sequence of events is generally 

1. Create a storage class
2. Create a PVC 
3. Create a pod or a replication controller that uses the PVC

```
# you can use the following command to create a storage class first
kubectl create -f storageclass-managed-hdd.json

# you can use the following command to create a pvc, which will create an azure disk
kubectl create -f pvc-on-managed-hdd.json

# You can get more details about the created PVC by
kubectl describe pvc {pvc-name}

# you can use the following command to create a pod with specified pvc
kubectl create -f pod-uses-managed-hdd.json
   
```

To verify, inside of the pod/container, you should see something like this:

```
$ df -h
/dev/sdc                125.9G     59.6M    119.4G   0% /mnt/managed
```



## How does it work? 

### Managed Disks
The entire experience is offloaded to Azure to manage disks:storage accounts. You can use PVC (Kubernetes will automatically create a managed disk for you). Or you can you use an existing disk as PV in your PODs/RCs

> as a general rule, use PV disks provisioned in the same Azure resource group where the cluster is provisioned.   

### Blob Disks 
Blob Disks works in two modes. Controlled by #kind# parameter on the storage class. 

### Dedicated (default mode)
When *kind* parameter is set to *dedicated* K8S will create a new dedicated storage account for this new disk. No other disks will be allowed in the this storage account. The account will removed when the PVC is removed (according to K8S PVC reclaim policy) 

> You can still use existing VHDs, again the general rule apply use storage accounts that are part of cluster resource group

### The following storage parameter can be used to control the behaviour

1. *skuname* or *storageaccounttype* to choose the underlying Azure storage account (default is *standard_lrs* allowed values are  *standard_lrs* and *premium_lrs*)
2. *cachingmode* controls Azure caching mode when the disk is attached to a VM (default is *readwrite* allowed values are *none*, *readwrite* and *readonly*
3. *kind* decides on disk kind (default is *shared* allowed values are *shared*, *dedicated* and *managed*)
4. *fstype* the file system of this disk (default *ext4*)

### Shared
PVC: VHDs are created in a shared storage accounts in the same resource group as the cluster as the following 

```
Rsource Group
--Storage Account: pvc{unique-hash}001 // created by K8S as it provisoned  PVC, all disks are placed in the same blob container  
---pvc-xxx-xxx-xxxx.vhd
---pvc-xxx-xxx-xxxx.vhd
--Storage Account: pvc{unique-hash}002..n  
---pvc-xxx-xxx-xxxx.vhd
```

The following rules apply:

1. Maximum # of accounts created by K8S for shared PVC is **100**.
2. Maximum # of disks per account is 60 (VHDs).
3. K8S will create new account for new disks if * utilization(AccountType(NEWDISK)) > 50%  * keeping total # of accounts below 100.
4. K8S will create initial 2 accounts ( 1 standard and 1 premium ) to accelerate the provisioning process.

## Additional Notes
The samples assumes that you have a cluster with node labeled with #disktype=blob# for VMs that are using blob disks and #disktype=managed# for VMs that are using managed disks. You can label your nodes or remove the node selector before using the files. 

> You can not attach managed disks to VMs that are not using managed OS disks. This applies also the other way around no blob disks on VMS that are using managed OS disks

To label your nodes use the following command 
```
kubectl label nodes {node-name-here} disktype=blob
```
