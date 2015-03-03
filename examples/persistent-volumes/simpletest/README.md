# How To Use Persistent Volumes

This guide assumes knowledge of Kubernetes fundamentals and that a user has a cluster up and running.

## Create volumes

Persistent Volumes are intended for "network volumes", such as GCE Persistent Disks, NFS shares, and AWS EBS volumes.

The `HostPath` VolumeSource was included in the Persistent Volumes implementation for ease of testing.
 
Create persistent volumes by posting them to the API server:

```

cluster/kubectl.sh create -f examples/persistent-volumes/volumes/local-01.yaml
cluster/kubectl.sh create -f examples/persistent-volumes/volumes/local-02.yaml

cluster/kubectl.sh get pv

NAME                LABELS              CAPACITY            ACCESSMODES         STATUS              CLAIM
pv0001              map[]               10737418240         RWO                                     
pv0002              map[]               5368709120          RWO        


In the log:

I0302 10:20:45.663225    1920 persistent_volume_manager.go:115] Managing PersistentVolume[UID=b16e91d6-c0ef-11e4-8be4-80e6500a981e]
I0302 10:20:55.667945    1920 persistent_volume_manager.go:115] Managing PersistentVolume[UID=b41f4f0e-c0ef-11e4-8be4-80e6500a981e]

```

## Create claims

You must be in a namespace to create claims.

```

cluster/kubectl.sh create -f examples/persistent-volumes/claims/claim-01.yaml
cluster/kubectl.sh create -f examples/persistent-volumes/claims/claim-02.yaml

NAME                LABELS              STATUS              VOLUME
myclaim-1           map[]                                   
myclaim-2           map[]                                   

```


## Matching and binding

```

PersistentVolumeClaim[UID=f4b3d283-c0ef-11e4-8be4-80e6500a981e] bound to PersistentVolume[UID=b16e91d6-c0ef-11e4-8be4-80e6500a981e]



cluster/kubectl.sh get pv

NAME                LABELS              CAPACITY            ACCESSMODES         STATUS              CLAIM
pv0001              map[]               10737418240         RWO                                     myclaim-1 / f4b3d283-c0ef-11e4-8be4-80e6500a981e
pv0002              map[]               5368709120          RWO                                     myclaim-2 / f70da891-c0ef-11e4-8be4-80e6500a981e


cluster/kubectl.sh get pvc

NAME                LABELS              STATUS              VOLUME
myclaim-1           map[]                                   b16e91d6-c0ef-11e4-8be4-80e6500a981e
myclaim-2           map[]                                   b41f4f0e-c0ef-11e4-8be4-80e6500a981e

```
