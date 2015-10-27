<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# How to Use it?

Install Ceph on the Kubernetes host. For example, on Fedora 21

    # yum -y install ceph

If you don't have a Ceph cluster, you can set up a [containerized Ceph cluster](https://github.com/rootfs/ceph_docker)

Then get the keyring from the Ceph cluster and copy it to */etc/ceph/keyring*.

Once you have installed Ceph and new Kubernetes, you can create a pod based on my examples [cephfs.json](cephfs.json)  and [cephfs-with-secret.json](cephfs-with-secret.json). In the pod JSON, you need to provide the following information.

- *monitors*:  Array of Ceph monitors.
- *user*: The RADOS user name. If not provided, default *admin* is used.
- *secretFile*: The path to the keyring file. If not provided, default */etc/ceph/user.secret* is used.
- *secretRef*: Reference to Ceph authentication secrets. If provided, *secret* overrides *secretFile*.
- *readOnly*: Whether the filesystem is used as readOnly.


Here are the commands:

```console
    # create a secret if you want to use Ceph secret instead of secret file
    # cluster/kubectl.sh create -f examples/cephfs/secret/ceph-secret.yaml
	
    # cluster/kubectl.sh create -f examples/cephfs/v1beta3/cephfs.json
    # cluster/kubectl.sh get pods
```

 If you ssh to that machine, you can run `docker ps` to see the actual pod and `docker inspect` to see the volumes used by the container.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cephfs/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
