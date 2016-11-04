# MySQL installation with cinder volume plugin

Cinder is a Block Storage service for OpenStack. This example shows how it can be used as an attachment mounted to a pod in Kubernets.

### Prerequisites

Start kubelet with cloud provider as openstack with a valid cloud config
Sample cloud_config:

```
[Global]
auth-url=https://os-identity.vip.foo.bar.com:5443/v2.0
username=user
password=pass
region=region1
tenant-id=0c331a1df18571594d49fe68asa4e
```

Currently the cinder volume plugin is designed to work only on linux hosts and offers ext4 and ext3 as supported fs types
Make sure that kubelet host machine has the following executables

```
/bin/lsblk -- To Find out the fstype of the volume
/sbin/mkfs.ext3 and /sbin/mkfs.ext4 -- To format the volume if required
/usr/bin/udevadm -- To probe the volume attached so that a symlink is created under /dev/disk/by-id/ with a virtio- prefix
```

Ensure cinder is installed and configured properly in the region in which kubelet is spun up

### Example

Create a cinder volume Ex:

`cinder create --display-name=test-repo 2`

Use the id of the cinder volume created to create a pod [definition](mysql.yaml)
Create a new pod with the definition

`cluster/kubectl.sh create -f examples/mysql-cinder-pd/mysql.yaml`

This should now

1. Attach the specified volume to the kubelet's host machine
2. Format the volume if required (only if the volume specified is not already formatted to the fstype specified)
3. Mount it on the kubelet's host machine
4. Spin up a container with this volume mounted to the path specified in the pod definition


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/mysql-cinder-pd/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
