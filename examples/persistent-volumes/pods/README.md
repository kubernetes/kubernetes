# How To Use Persistent Volumes

This guide assumes knowledge of Kubernetes fundamentals and that a user has a cluster up and running.

## Step 1: Admin provisions volumes

Persistent Volumes are intended for "network volumes", such as GCE Persistent Disks, NFS shares, and AWS EBS volumes.

The `HostPath` VolumeSource was included explicitly for testing.  There is no on-host persistent storage at this time.
 
Create persistent volumes by posting them to the API server:

```

cluster/kubectl.sh create -f examples/persistent-volumes/volumes/local-01.yaml
cluster/kubectl.sh get pv

NAME                LABELS              CAPACITY            ACCESSMODES         STATUS              CLAIM
pv0001              map[]               10737418240         RWO                                     
pv0002              map[]               5368709120          RWO        


In the log:

I0302 10:20:45.663225    1920 persistent_volume_manager.go:115] Managing PersistentVolume[UID=b16e91d6-c0ef-11e4-8be4-80e6500a981e]
I0302 10:20:55.667945    1920 persistent_volume_manager.go:115] Managing PersistentVolume[UID=b41f4f0e-c0ef-11e4-8be4-80e6500a981e]

```

## Step 2: Request a volume

You must be in a namespace to create claims.

```

cluster/kubectl.sh create -f examples/persistent-volumes/claims/claim-01.yaml
cluster/kubectl.sh create -f examples/persistent-volumes/claims/claim-02.yaml


cluster/kubectl.sh get pvc

NAME                LABELS              STATUS              VOLUME
myclaim-1           map[]                                   b16e91d6-c0ef-11e4-8be4-80e6500a981e
myclaim-2           map[]                                   b41f4f0e-c0ef-11e4-8be4-80e6500a981e

```


## Step 3: Use Claim as volume

Use the claim as a volume in a pod.


```

cluster/kubectl.sh create -f examples/persistent-volumes/pods/pod.yaml

$ cluster/kubectl.sh get pod
Running: cluster/../cluster/gce/../../_output/local/bin/linux/amd64/kubectl --v=5 get pod
POD                 IP                  CONTAINER(S)        IMAGE(S)            HOST                LABELS              STATUS              CREATED
mypod                                   myfrontend          dockerfile/nginx    127.0.0.1/          name=frontendhttp   Pending             4 seconds

```



## Step 4:  Create service


```

cluster/kubectl.sh create -f examples/persistent-volumes/pods/service.yaml

$ cluster/kubectl.sh get service
Running: cluster/../cluster/gce/../../_output/local/bin/linux/amd64/kubectl --v=5 get service
NAME                LABELS                                    SELECTOR            IP                  PORT
frontendservice     <none>                                    name=frontendhttp   10.0.0.114          3000
kubernetes          component=apiserver,provider=kubernetes   <none>              10.0.0.2            443
kubernetes-ro       component=apiserver,provider=kubernetes   <none>              10.0.0.1            80



$ curl 10.0.0.114:3000
This is content from /tmp/data02/index.html!!!!  woot!

```

