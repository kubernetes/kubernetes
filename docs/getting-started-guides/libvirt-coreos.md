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
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/libvirt-coreos.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Getting started with libvirt CoreOS
-----------------------------------

**Table of Contents**

- [Highlights](#highlights)
- [Warnings about `libvirt-coreos` use case](#warnings-about-libvirt-coreos-use-case)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Interacting with your Kubernetes cluster with the `kube-*` scripts.](#interacting-with-your-kubernetes-cluster-with-the-kube--scripts)
- [Troubleshooting](#troubleshooting)
    - [!!! Cannot find kubernetes-server-linux-amd64.tar.gz](#-cannot-find-kubernetes-server-linux-amd64targz)
    - [Can't find virsh in PATH, please fix and retry.](#cant-find-virsh-in-path-please-fix-and-retry)
    - [error: Failed to connect socket to '/var/run/libvirt/libvirt-sock': No such file or directory](#error-failed-to-connect-socket-to-varrunlibvirtlibvirt-sock-no-such-file-or-directory)
    - [error: Failed to connect socket to '/var/run/libvirt/libvirt-sock': Permission denied](#error-failed-to-connect-socket-to-varrunlibvirtlibvirt-sock-permission-denied)
    - [error: Out of memory initializing network (virsh net-create...)](#error-out-of-memory-initializing-network-virsh-net-create)

### Highlights

* Super-fast cluster boot-up (few seconds instead of several minutes for vagrant)
* Reduced disk usage thanks to [COW](https://en.wikibooks.org/wiki/QEMU/Images#Copy_on_write)
* Reduced memory footprint thanks to [KSM](https://www.kernel.org/doc/Documentation/vm/ksm.txt)

### Warnings about `libvirt-coreos` use case

The primary goal of the `libvirt-coreos` cluster provider is to deploy a multi-node Kubernetes cluster on local VMs as fast as possible and to be as light as possible in term of resources used.

In order to achieve that goal, its deployment is very different from the “standard production deployment” method used on other providers. This was done on purpose in order to implement some optimizations made possible by the fact that we know that all VMs will be running on the same physical machine.

The `libvirt-coreos` cluster provider doesn’t aim at being production look-alike.

Another difference is that no security is enforced on `libvirt-coreos` at all. For example,

* Kube API server is reachable via a clear-text connection (no SSL);
* Kube API server requires no credentials;
* etcd access is not protected;
* Kubernetes secrets are not protected as securely as they are on production environments;
* etc.

So, an k8s application developer should not validate its interaction with Kubernetes on `libvirt-coreos` because he might technically succeed in doing things that are prohibited on a production environment like:

* un-authenticated access to Kube API server;
* Access to Kubernetes private data structures inside etcd;
* etc.

On the other hand, `libvirt-coreos` might be useful for people investigating low level implementation of Kubernetes because debugging techniques like sniffing the network traffic or introspecting the etcd content are easier on `libvirt-coreos` than on a production deployment.

### Prerequisites

1. Install [dnsmasq](http://www.thekelleys.org.uk/dnsmasq/doc.html)
2. Install [ebtables](http://ebtables.netfilter.org/)
3. Install [qemu](http://wiki.qemu.org/Main_Page)
4. Install [libvirt](http://libvirt.org/)
5. Enable and start the libvirt daemon, e.g:
   * ``systemctl enable libvirtd``
   * ``systemctl start libvirtd``
6. [Grant libvirt access to your user¹](https://libvirt.org/aclpolkit.html)
7. Check that your $HOME is accessible to the qemu user²

#### ¹ Depending on your distribution, libvirt access may be denied by default or may require a password at each access.

You can test it with the following command:

```sh
virsh -c qemu:///system pool-list
```

If you have access error messages, please read https://libvirt.org/acl.html and https://libvirt.org/aclpolkit.html .

In short, if your libvirt has been compiled with Polkit support (ex: Arch, Fedora 21), you can create `/etc/polkit-1/rules.d/50-org.libvirt.unix.manage.rules` as follows to grant full access to libvirt to `$USER`

```sh
sudo /bin/sh -c "cat - > /etc/polkit-1/rules.d/50-org.libvirt.unix.manage.rules" << EOF
polkit.addRule(function(action, subject) {
        if (action.id == "org.libvirt.unix.manage" &&
            subject.user == "$USER") {
                return polkit.Result.YES;
                polkit.log("action=" + action);
                polkit.log("subject=" + subject);
        }
});
EOF
```

If your libvirt has not been compiled with Polkit (ex: Ubuntu 14.04.1 LTS), check the permissions on the libvirt unix socket:

```console
$ ls -l /var/run/libvirt/libvirt-sock
srwxrwx--- 1 root libvirtd 0 févr. 12 16:03 /var/run/libvirt/libvirt-sock

$ usermod -a -G libvirtd $USER
# $USER needs to logout/login to have the new group be taken into account
```

(Replace `$USER` with your login name)

#### ² Qemu will run with a specific user. It must have access to the VMs drives

All the disk drive resources needed by the VM (CoreOS disk image, Kubernetes binaries, cloud-init files, etc.) are put inside `./cluster/libvirt-coreos/libvirt_storage_pool`.

As we’re using the `qemu:///system` instance of libvirt, qemu will run with a specific `user:group` distinct from your user. It is configured in `/etc/libvirt/qemu.conf`. That qemu user must have access to that libvirt storage pool.

If your `$HOME` is world readable, everything is fine. If your $HOME is private, `cluster/kube-up.sh` will fail with an error message like:

```console
error: Cannot access storage file '$HOME/.../kubernetes/cluster/libvirt-coreos/libvirt_storage_pool/kubernetes_master.img' (as uid:99, gid:78): Permission denied
```

In order to fix that issue, you have several possibilities:
* set `POOL_PATH` inside `cluster/libvirt-coreos/config-default.sh` to a directory:
  * backed by a filesystem with a lot of free disk space
  * writable by your user;
  * accessible by the qemu user.
* Grant the qemu user access to the storage pool.

On Arch:

```sh
setfacl -m g:kvm:--x ~
```

### Setup

By default, the libvirt-coreos setup will create a single Kubernetes master and 3 Kubernetes nodes. Because the VM drives use Copy-on-Write and because of memory ballooning and KSM, there is a lot of resource over-allocation.

To start your local cluster, open a shell and run:

```sh
cd kubernetes

export KUBERNETES_PROVIDER=libvirt-coreos
cluster/kube-up.sh
```

The `KUBERNETES_PROVIDER` environment variable tells all of the various cluster management scripts which variant to use.  If you forget to set this, the assumption is you are running on Google Compute Engine.

The `NUM_MINIONS` environment variable may be set to specify the number of nodes to start. If it is not set, the number of nodes defaults to 3.

The `KUBE_PUSH` environment variable may be set to specify which Kubernetes binaries must be deployed on the cluster. Its possible values are:

* `release` (default if `KUBE_PUSH` is not set) will deploy the binaries of `_output/release-tars/kubernetes-server-….tar.gz`. This is built with `make release` or `make release-skip-tests`.
* `local` will deploy the binaries of `_output/local/go/bin`. These are built with `make`.

You can check that your machines are there and running with:

```console
$ virsh -c qemu:///system list
 Id    Name                           State
----------------------------------------------------
 15    kubernetes_master              running
 16    kubernetes_minion-01           running
 17    kubernetes_minion-02           running
 18    kubernetes_minion-03           running
 ```

You can check that the Kubernetes cluster is working with:

```console
$ kubectl get nodes
NAME                LABELS              STATUS
192.168.10.2        <none>              Ready
192.168.10.3        <none>              Ready
192.168.10.4        <none>              Ready
```

The VMs are running [CoreOS](https://coreos.com/).
Your ssh keys have already been pushed to the VM. (It looks for ~/.ssh/id_*.pub)
The user to use to connect to the VM is `core`.
The IP to connect to the master is 192.168.10.1.
The IPs to connect to the nodes are 192.168.10.2 and onwards.

Connect to `kubernetes_master`:

```sh
ssh core@192.168.10.1
```

Connect to `kubernetes_minion-01`:

```sh
ssh core@192.168.10.2
```

### Interacting with your Kubernetes cluster with the `kube-*` scripts.

All of the following commands assume you have set `KUBERNETES_PROVIDER` appropriately:

```sh
export KUBERNETES_PROVIDER=libvirt-coreos
```

Bring up a libvirt-CoreOS cluster of 5 nodes

```sh
NUM_MINIONS=5 cluster/kube-up.sh
```

Destroy the libvirt-CoreOS cluster

```sh
cluster/kube-down.sh
```

Update the libvirt-CoreOS cluster with a new Kubernetes release produced by `make release` or `make release-skip-tests`:

```sh
cluster/kube-push.sh
```

Update the libvirt-CoreOS cluster with the locally built Kubernetes binaries produced by `make`:

```sh
KUBE_PUSH=local cluster/kube-push.sh
```

Interact with the cluster

```sh
kubectl ...
```

### Troubleshooting

#### !!! Cannot find kubernetes-server-linux-amd64.tar.gz

Build the release tarballs:

```sh
make release
```

#### Can't find virsh in PATH, please fix and retry.

Install libvirt

On Arch:

```sh
pacman -S qemu libvirt
```

On Ubuntu 14.04.1:

```sh
aptitude install qemu-system-x86 libvirt-bin
```

On Fedora 21:

```sh
yum install qemu libvirt
```

#### error: Failed to connect socket to '/var/run/libvirt/libvirt-sock': No such file or directory

Start the libvirt daemon

On Arch:

```sh
systemctl start libvirtd
```

On Ubuntu 14.04.1:

```sh
service libvirt-bin start
```

#### error: Failed to connect socket to '/var/run/libvirt/libvirt-sock': Permission denied

Fix libvirt access permission (Remember to adapt `$USER`)

On Arch and Fedora 21:

```sh
cat > /etc/polkit-1/rules.d/50-org.libvirt.unix.manage.rules <<EOF
polkit.addRule(function(action, subject) {
        if (action.id == "org.libvirt.unix.manage" &&
            subject.user == "$USER") {
                return polkit.Result.YES;
                polkit.log("action=" + action);
                polkit.log("subject=" + subject);
        }
});
EOF
```

On Ubuntu:

```sh
usermod -a -G libvirtd $USER
```

#### error: Out of memory initializing network (virsh net-create...)

Ensure libvirtd has been restarted since ebtables was installed.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/libvirt-coreos.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
