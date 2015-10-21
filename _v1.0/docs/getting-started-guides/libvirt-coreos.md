---
layout: docwithnav
title: "Getting started with libvirt CoreOS"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
Getting started with libvirt CoreOS
-----------------------------------

**Table of Contents**

- [Highlights](#highlights)
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

{% highlight sh %}
{% raw %}
virsh -c qemu:///system pool-list
{% endraw %}
{% endhighlight %}

If you have access error messages, please read https://libvirt.org/acl.html and https://libvirt.org/aclpolkit.html .

In short, if your libvirt has been compiled with Polkit support (ex: Arch, Fedora 21), you can create `/etc/polkit-1/rules.d/50-org.libvirt.unix.manage.rules` as follows to grant full access to libvirt to `$USER`

{% highlight sh %}
{% raw %}
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
{% endraw %}
{% endhighlight %}

If your libvirt has not been compiled with Polkit (ex: Ubuntu 14.04.1 LTS), check the permissions on the libvirt unix socket:

{% highlight console %}
{% raw %}
$ ls -l /var/run/libvirt/libvirt-sock
srwxrwx--- 1 root libvirtd 0 févr. 12 16:03 /var/run/libvirt/libvirt-sock

$ usermod -a -G libvirtd $USER
# $USER needs to logout/login to have the new group be taken into account
{% endraw %}
{% endhighlight %}

(Replace `$USER` with your login name)

#### ² Qemu will run with a specific user. It must have access to the VMs drives

All the disk drive resources needed by the VM (CoreOS disk image, Kubernetes binaries, cloud-init files, etc.) are put inside `./cluster/libvirt-coreos/libvirt_storage_pool`.

As we’re using the `qemu:///system` instance of libvirt, qemu will run with a specific `user:group` distinct from your user. It is configured in `/etc/libvirt/qemu.conf`. That qemu user must have access to that libvirt storage pool.

If your `$HOME` is world readable, everything is fine. If your $HOME is private, `cluster/kube-up.sh` will fail with an error message like:

{% highlight console %}
{% raw %}
error: Cannot access storage file '$HOME/.../kubernetes/cluster/libvirt-coreos/libvirt_storage_pool/kubernetes_master.img' (as uid:99, gid:78): Permission denied
{% endraw %}
{% endhighlight %}

In order to fix that issue, you have several possibilities:
* set `POOL_PATH` inside `cluster/libvirt-coreos/config-default.sh` to a directory:
  * backed by a filesystem with a lot of free disk space
  * writable by your user;
  * accessible by the qemu user.
* Grant the qemu user access to the storage pool.

On Arch:

{% highlight sh %}
{% raw %}
setfacl -m g:kvm:--x ~
{% endraw %}
{% endhighlight %}

### Setup

By default, the libvirt-coreos setup will create a single Kubernetes master and 3 Kubernetes nodes. Because the VM drives use Copy-on-Write and because of memory ballooning and KSM, there is a lot of resource over-allocation.

To start your local cluster, open a shell and run:

{% highlight sh %}
{% raw %}
cd kubernetes

export KUBERNETES_PROVIDER=libvirt-coreos
cluster/kube-up.sh
{% endraw %}
{% endhighlight %}

The `KUBERNETES_PROVIDER` environment variable tells all of the various cluster management scripts which variant to use.  If you forget to set this, the assumption is you are running on Google Compute Engine.

The `NUM_MINIONS` environment variable may be set to specify the number of nodes to start. If it is not set, the number of nodes defaults to 3.

The `KUBE_PUSH` environment variable may be set to specify which Kubernetes binaries must be deployed on the cluster. Its possible values are:

* `release` (default if `KUBE_PUSH` is not set) will deploy the binaries of `_output/release-tars/kubernetes-server-….tar.gz`. This is built with `make release` or `make release-skip-tests`.
* `local` will deploy the binaries of `_output/local/go/bin`. These are built with `make`.

You can check that your machines are there and running with:

{% highlight console %}
{% raw %}
$ virsh -c qemu:///system list
 Id    Name                           State
----------------------------------------------------
 15    kubernetes_master              running
 16    kubernetes_minion-01           running
 17    kubernetes_minion-02           running
 18    kubernetes_minion-03           running
{% endraw %}
{% endhighlight %}

You can check that the Kubernetes cluster is working with:

{% highlight console %}
{% raw %}
$ kubectl get nodes
NAME                LABELS              STATUS
192.168.10.2        <none>              Ready
192.168.10.3        <none>              Ready
192.168.10.4        <none>              Ready
{% endraw %}
{% endhighlight %}

The VMs are running [CoreOS](https://coreos.com/).
Your ssh keys have already been pushed to the VM. (It looks for ~/.ssh/id_*.pub)
The user to use to connect to the VM is `core`.
The IP to connect to the master is 192.168.10.1.
The IPs to connect to the nodes are 192.168.10.2 and onwards.

Connect to `kubernetes_master`:

{% highlight sh %}
{% raw %}
ssh core@192.168.10.1
{% endraw %}
{% endhighlight %}

Connect to `kubernetes_minion-01`:

{% highlight sh %}
{% raw %}
ssh core@192.168.10.2
{% endraw %}
{% endhighlight %}

### Interacting with your Kubernetes cluster with the `kube-*` scripts.

All of the following commands assume you have set `KUBERNETES_PROVIDER` appropriately:

{% highlight sh %}
{% raw %}
export KUBERNETES_PROVIDER=libvirt-coreos
{% endraw %}
{% endhighlight %}

Bring up a libvirt-CoreOS cluster of 5 nodes

{% highlight sh %}
{% raw %}
NUM_MINIONS=5 cluster/kube-up.sh
{% endraw %}
{% endhighlight %}

Destroy the libvirt-CoreOS cluster

{% highlight sh %}
{% raw %}
cluster/kube-down.sh
{% endraw %}
{% endhighlight %}

Update the libvirt-CoreOS cluster with a new Kubernetes release produced by `make release` or `make release-skip-tests`:

{% highlight sh %}
{% raw %}
cluster/kube-push.sh
{% endraw %}
{% endhighlight %}

Update the libvirt-CoreOS cluster with the locally built Kubernetes binaries produced by `make`:

{% highlight sh %}
{% raw %}
KUBE_PUSH=local cluster/kube-push.sh
{% endraw %}
{% endhighlight %}

Interact with the cluster

{% highlight sh %}
{% raw %}
kubectl ...
{% endraw %}
{% endhighlight %}

### Troubleshooting

#### !!! Cannot find kubernetes-server-linux-amd64.tar.gz

Build the release tarballs:

{% highlight sh %}
{% raw %}
make release
{% endraw %}
{% endhighlight %}

#### Can't find virsh in PATH, please fix and retry.

Install libvirt

On Arch:

{% highlight sh %}
{% raw %}
pacman -S qemu libvirt
{% endraw %}
{% endhighlight %}

On Ubuntu 14.04.1:

{% highlight sh %}
{% raw %}
aptitude install qemu-system-x86 libvirt-bin
{% endraw %}
{% endhighlight %}

On Fedora 21:

{% highlight sh %}
{% raw %}
yum install qemu libvirt
{% endraw %}
{% endhighlight %}

#### error: Failed to connect socket to '/var/run/libvirt/libvirt-sock': No such file or directory

Start the libvirt daemon

On Arch:

{% highlight sh %}
{% raw %}
systemctl start libvirtd
{% endraw %}
{% endhighlight %}

On Ubuntu 14.04.1:

{% highlight sh %}
{% raw %}
service libvirt-bin start
{% endraw %}
{% endhighlight %}

#### error: Failed to connect socket to '/var/run/libvirt/libvirt-sock': Permission denied

Fix libvirt access permission (Remember to adapt `$USER`)

On Arch and Fedora 21:

{% highlight sh %}
{% raw %}
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
{% endraw %}
{% endhighlight %}

On Ubuntu:

{% highlight sh %}
{% raw %}
usermod -a -G libvirtd $USER
{% endraw %}
{% endhighlight %}

#### error: Out of memory initializing network (virsh net-create...)

Ensure libvirtd has been restarted since ebtables was installed.


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/libvirt-coreos.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

