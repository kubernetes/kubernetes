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

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/examples/iscsi/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Step 1. Setting up iSCSI target and iSCSI initiator

**Setup A.** On Fedora 21 nodes

If you use Fedora 21 on Kubernetes node, then first install iSCSI initiator on the node:

    # yum -y install iscsi-initiator-utils


then edit */etc/iscsi/initiatorname.iscsi* and */etc/iscsi/iscsid.conf* to match your iSCSI target configuration.

I mostly followed these [instructions](http://www.server-world.info/en/note?os=Fedora_21&p=iscsi) to setup iSCSI target. and these [instructions](http://www.server-world.info/en/note?os=Fedora_21&p=iscsi&f=2) to setup iSCSI initiator.

**Setup B.** On Unbuntu 12.04 and Debian 7 nodes on Google Compute Engine (GCE)

GCE does not provide preconfigured Fedora 21 image, so I set up the iSCSI target on a preconfigured Ubuntu 12.04 image, mostly following these [instructions](http://www.server-world.info/en/note?os=Ubuntu_12.04&p=iscsi). My Kubernetes cluster on GCE was running Debian 7 images, so I followed these [instructions](http://www.server-world.info/en/note?os=Debian_7.0&p=iscsi&f=2) to set up the iSCSI initiator.

## Step 2. Creating the pod with iSCSI persistent storage

Once you have installed iSCSI initiator and new Kubernetes, you can create a pod based on my example *iscsi.json*. In the pod JSON, you need to provide *targetPortal* (the iSCSI target's **IP** address and *port* if not the default port 3260), target's *iqn*, *lun*, and the type of the filesystem that has been created on the lun, and *readOnly* boolean.

**Note:** If you have followed the instructions in the links above you
may have partitioned the device, the iSCSI volume plugin does not
currently support partitions so format the device as one partition.
Make sure you have the correct device name then run the following as
root to format it:

```console
mkfs.ext4 /dev/<name of device>
```

Once your pod is created, run it on the Kubernetes master:

```console
kubectl create -f ./your_new_pod.json
```

Here is my command and output:

```console
# kubectl create -f examples/iscsi/iscsi.json
# kubectl get pods
NAME      READY     STATUS    RESTARTS   AGE
iscsipd   2/2       RUNNING   0           2m
```

On the Kubernetes node, I got these in mount output

```console
# mount |grep kub
/dev/sdb on /var/lib/kubelet/plugins/kubernetes.io/iscsi/iscsi/10.240.205.13:3260-iqn-iqn.2014-12.world.server:storage.target1-lun-0 type ext4 (ro,relatime,data=ordered)
/dev/sdb on /var/lib/kubelet/pods/e36158ce-f8d8-11e4-9ae7-42010af01964/volumes/kubernetes.io~iscsi/iscsipd-ro type ext4 (ro,relatime,data=ordered)
/dev/sdc on /var/lib/kubelet/plugins/kubernetes.io/iscsi/iscsi/10.240.205.13:3260-iqn-iqn.2014-12.world.server:storage.target1-lun-1 type xfs (rw,relatime,attr2,inode64,noquota)
/dev/sdc on /var/lib/kubelet/pods/e36158ce-f8d8-11e4-9ae7-42010af01964/volumes/kubernetes.io~iscsi/iscsipd-rw type xfs (rw,relatime,attr2,inode64,noquota)
```

If you ssh to that machine, you can run `docker ps` to see the actual pod.

```console
# docker ps
CONTAINER ID        IMAGE                                                COMMAND                CREATED             STATUS              PORTS               NAMES
cc051196e7af        kubernetes/pause:latest                              "/pause"               About an hour ago   Up About an hour                        k8s_iscsipd-rw.ff2d2e9f_iscsipd_default_e36158ce-f8d8-11e4-9ae7-42010af01964_26f3a457                                               
8aa981443cf4        kubernetes/pause:latest                              "/pause"               About an hour ago   Up About an hour                        k8s_iscsipd-ro.d7752e8f_iscsipd_default_e36158ce-f8d8-11e4-9ae7-42010af01964_4939633d    
```

Run *docker inspect* and I found the Containers mounted the host directory into the their */mnt/iscsipd* directory.

```console 
# docker inspect --format '{{index .Volumes "/mnt/iscsipd"}}' cc051196e7af
/var/lib/kubelet/pods/75e0af2b-f8e8-11e4-9ae7-42010af01964/volumes/kubernetes.io~iscsi/iscsipd-rw
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/iscsi/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
