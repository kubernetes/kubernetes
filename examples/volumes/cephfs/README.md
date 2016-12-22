# How to Use it?

Install Ceph on the Kubernetes host. For example, on Fedora 21

    # yum -y install ceph

If you don't have a Ceph cluster, you can set up a [containerized Ceph cluster](https://github.com/ceph/ceph-docker/tree/master/examples/kubernetes)

Then get the keyring from the Ceph cluster and copy it to */etc/ceph/keyring*.

Once you have installed Ceph and a Kubernetes cluster, you can create a pod based on my examples [cephfs.yaml](cephfs.yaml)  and [cephfs-with-secret.yaml](cephfs-with-secret.yaml). In the pod yaml, you need to provide the following information.

- *monitors*:  Array of Ceph monitors.
- *path*: Used as the mounted root, rather than the full Ceph tree. If not provided, default */* is used.
- *user*: The RADOS user name. If not provided, default *admin* is used.
- *secretFile*: The path to the keyring file. If not provided, default */etc/ceph/user.secret* is used.
- *secretRef*: Reference to Ceph authentication secrets. If provided, *secret* overrides *secretFile*.
- *readOnly*: Whether the filesystem is used as readOnly.


Here are the commands:

```console
    # kubectl create -f examples/volumes/cephfs/cephfs.yaml

    # create a secret if you want to use Ceph secret instead of secret file
    # kubectl create -f examples/volumes/cephfs/secret/ceph-secret.yaml
	
    # kubectl create -f examples/volumes/cephfs/cephfs-with-secret.yaml
    # kubectl get pods
```

 If you ssh to that machine, you can run `docker ps` to see the actual pod and `docker inspect` to see the volumes used by the container.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/cephfs/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
