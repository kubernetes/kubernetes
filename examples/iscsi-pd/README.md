# How to Use it?
Here is my setup to setup Kubernetes with iSCSI persistent storage. I use Fedora 21 on Kubernetes node. 

Install iSCSI initiator on the node:

    # yum -y install iscsi-initiator-utils
   
   
then edit */etc/iscsi/initiatorname.iscsi* and */etc/iscsi/iscsid.conf* to match your iSCSI target configuration.

I mostly follow these [instructions](http://www.server-world.info/en/note?os=Fedora_21&p=iscsi&f=2) to setup iSCSI initiator and these [instructions](http://www.server-world.info/en/note?os=Fedora_21&p=iscsi) to setup iSCSI target.

Once you have installed iSCSI initiator and new Kubernetes, you can create a pod based on my example *iscsi-pd.json*. In the pod JSON, you need to provide *portal* (the iSCSI target's **IP** address and *port* if not the default port 3260), target's *iqn*, *lun*, and the type of the filesystem that has been created on the lun, and *readOnly* boolean. 

Once your pod is created, run it on the Kubernetes master:

    #cluster/kubectl.sh create -f your_new_pod.json

Here is my command and output:

```console
    # cluster/kubectl.sh create -f examples/iscsi-pd/iscsi-pd.json 
    current-context: ""
    Running: cluster/../cluster/gce/../../_output/local/bin/linux/amd64/kubectl create -f examples/iscsi-pd/iscsi-pd.json
    iscsipd
    # cluster/kubectl.sh get pods
    current-context: ""
    Running: cluster/../cluster/gce/../../_output/local/bin/linux/amd64/kubectl get pods
    POD                                    IP                  CONTAINER(S)        IMAGE(S)                 HOST                      LABELS              STATUS
    iscsipd                                172.17.0.6          iscsipd-ro          kubernetes/pause         fed-minion/10.16.154.75   <none>              Running
                                                           iscsipd-rw          kubernetes/pause                                                    
```

On the Kubernetes node, I got these in mount output

```console
    #mount |grep kub
    /dev/sdb on /var/lib/kubelet/plugins/kubernetes.io/iscsi-pd/iscsi/10.16.154.81:3260/iqn.2014-12.world.server:storage.target1/lun/0 type ext4 (ro,relatime,stripe=1024,data=ordered)
    /dev/sdb on /var/lib/kubelet/pods/4ab78fdc-b927-11e4-ade6-d4bed9b39058/volumes/kubernetes.io~iscsi-pd/iscsipd-ro type ext4 (ro,relatime,stripe=1024,data=ordered)
    /dev/sdc on /var/lib/kubelet/plugins/kubernetes.io/iscsi-pd/iscsi/10.16.154.81:3260/iqn.2014-12.world.server:storage.target1/lun/1 type xfs (rw,relatime,attr2,inode64,noquota)
    /dev/sdc on /var/lib/kubelet/pods/4ab78fdc-b927-11e4-ade6-d4bed9b39058/volumes/kubernetes.io~iscsi-pd/iscsipd-rw type xfs (rw,relatime,attr2,inode64,noquota)
```

 Run *docker inspect* and I found the Containers mounted the host directory into the their */mnt/iscsipd* directory.
 
 ```console
    # docker ps
    CONTAINER ID        IMAGE                     COMMAND                CREATED             STATUS              PORTS                    NAMES
    cc9bd22d9e9d        kubernetes/pause:latest   "/pause"               3 minutes ago       Up 3 minutes                                 k8s_iscsipd-rw.12d8f0c5_iscsipd.default.etcd_4ab78fdc-b927-11e4-ade6-d4bed9b39058_e3f49dcc                               
    a4225a2148e3        kubernetes/pause:latest   "/pause"               3 minutes ago       Up 3 minutes                                 k8s_iscsipd-ro.f3c9f0b5_iscsipd.default.etcd_4ab78fdc-b927-11e4-ade6-d4bed9b39058_3cc9946f                               
    4d926d8989b3        kubernetes/pause:latest   "/pause"               3 minutes ago       Up 3 minutes                                 k8s_POD.8149c85a_iscsipd.default.etcd_4ab78fdc-b927-11e4-ade6-d4bed9b39058_c7b55d86                                      
    #docker inspect --format "\{\{\.Volumes\}\}" cc9bd22d9e9d
    map[/mnt/iscsipd:/var/lib/kubelet/pods/4ab78fdc-b927-11e4-ade6-d4bed9b39058/volumes/kubernetes.io~iscsi-pd/iscsipd-rw /dev/termination-log:/var/lib/kubelet/pods/4ab78fdc-b927-11e4-ade6-d4bed9b39058/containers/iscsipd-rw/cc9bd22d9e9db3c88a150cadfdccd86e36c463629035b48bdcfc8ec534be8615]
    #docker inspect --format "\{\{\.Volumes\}\}" a4225a2148e3
    map[/dev/termination-log:/var/lib/kubelet/pods/4ab78fdc-b927-11e4-ade6-d4bed9b39058/containers/iscsipd-ro/a4225a2148e38afc1a50a540ea9fe2e747886f1011ac5b3be4badee938f2fc5f /mnt/iscsipd:/var/lib/kubelet/pods/4ab78fdc-b927-11e4-ade6-d4bed9b39058/volumes/kubernetes.io~iscsi-pd/iscsipd-ro]
```