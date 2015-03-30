## Getting started with libvirt CoreOS

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
```
virsh -c qemu:///system pool-list
```

If you have access error messages, please read https://libvirt.org/acl.html and https://libvirt.org/aclpolkit.html .

In short, if your libvirt has been compiled with Polkit support (ex: Arch, Fedora 21), you can create `/etc/polkit-1/rules.d/50-org.libvirt.unix.manage.rules` as follows to grant full access to libvirt to `$USER`

```
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

```
ls -l /var/run/libvirt/libvirt-sock
srwxrwx--- 1 root libvirtd 0 févr. 12 16:03 /var/run/libvirt/libvirt-sock

usermod -a -G libvirtd $USER
# $USER needs to logout/login to have the new group be taken into account
```

(Replace `$USER` with your login name)

#### ² Qemu will run with a specific user. It must have access to the VMs drives

All the disk drive resources needed by the VM (CoreOS disk image, lmktfy binaries, cloud-init files, etc.) are put inside `./cluster/libvirt-coreos/libvirt_storage_pool`.

As we’re using the `qemu:///system` instance of libvirt, qemu will run with a specific `user:group` distinct from your user. It is configured in `/etc/libvirt/qemu.conf`. That qemu user must have access to that libvirt storage pool.

If your `$HOME` is world readable, everything is fine. If your $HOME is private, `cluster/lmktfy-up.sh` will fail with an error message like:

```
error: Cannot access storage file '$HOME/.../lmktfy/cluster/libvirt-coreos/libvirt_storage_pool/lmktfy_master.img' (as uid:99, gid:78): Permission denied
```

In order to fix that issue, you have several possibilities:
* set `POOL_PATH` inside `cluster/libvirt-coreos/config-default.sh` to a directory:
  * backed by a filesystem with a lot of free disk space
  * writable by your user;
  * accessible by the qemu user.
* Grant the qemu user access to the storage pool.

On Arch:

```
setfacl -m g:kvm:--x ~
```

### Setup

By default, the libvirt-coreos setup will create a single lmktfy master and 3 lmktfy minions. Because the VM drives use Copy-on-Write and because of memory ballooning and KSM, there is a lot of resource over-allocation.

To start your local cluster, open a shell and run:

```shell
cd lmktfy

export LMKTFYRNETES_PROVIDER=libvirt-coreos
cluster/lmktfy-up.sh
```

The `LMKTFYRNETES_PROVIDER` environment variable tells all of the various cluster management scripts which variant to use.  If you forget to set this, the assumption is you are running on Google Compute Engine.

The `NUM_MINIONS` environment variable may be set to specify the number of minions to start. If it is not set, the number of minions defaults to 3.

The `LMKTFY_PUSH` environment variable may be set to specify which lmktfy binaries must be deployed on the cluster. Its possible values are:

* `release` (default if `LMKTFY_PUSH` is not set) will deploy the binaries of `_output/release-tars/lmktfy-server-….tar.gz`. This is built with `make release` or `make release-skip-tests`.
* `local` will deploy the binaries of `_output/local/go/bin`. These are built with `make`.

You can check that your machines are there and running with:

```
virsh -c qemu:///system list
 Id    Name                           State
----------------------------------------------------
 15    lmktfy_master              running
 16    lmktfy_minion-01           running
 17    lmktfy_minion-02           running
 18    lmktfy_minion-03           running
 ```

You can check that the lmktfy cluster is working with:

```
$ ./cluster/lmktfyctl.sh get minions
NAME                LABELS              STATUS
192.168.10.2        <none>              Ready
192.168.10.3        <none>              Ready
192.168.10.4        <none>              Ready
```

The VMs are running [CoreOS](https://coreos.com/).
Your ssh keys have already been pushed to the VM. (It looks for ~/.ssh/id_*.pub)
The user to use to connect to the VM is `core`.
The IP to connect to the master is 192.168.10.1.
The IPs to connect to the minions are 192.168.10.2 and onwards.

Connect to `lmktfy_master`:
```
ssh core@192.168.10.1
```

Connect to `lmktfy_minion-01`:
```
ssh core@192.168.10.2
```

### Interacting with your LMKTFY cluster with the `lmktfy-*` scripts.

All of the following commands assume you have set `LMKTFYRNETES_PROVIDER` appropriately:

```
export LMKTFYRNETES_PROVIDER=libvirt-coreos
```

Bring up a libvirt-CoreOS cluster of 5 minions

```
NUM_MINIONS=5 cluster/lmktfy-up.sh
```

Destroy the libvirt-CoreOS cluster

```
cluster/lmktfy-down.sh
```

Update the libvirt-CoreOS cluster with a new LMKTFY release produced by `make release` or `make release-skip-tests`:

```
cluster/lmktfy-push.sh
```

Update the libvirt-CoreOS cluster with the locally built LMKTFY binaries produced by `make`:
```
LMKTFY_PUSH=local cluster/lmktfy-push.sh
```

Interact with the cluster

```
cluster/lmktfyctl.sh
```

### Troubleshooting

#### !!! Cannot find lmktfy-server-linux-amd64.tar.gz

Build the release tarballs:

```
make release
```

#### Can't find virsh in PATH, please fix and retry.

Install libvirt

On Arch:

```
pacman -S qemu libvirt
```

On Ubuntu 14.04.1:

```
aptitude install qemu-system-x86 libvirt-bin
```

On Fedora 21:

```
yum install qemu libvirt
```

#### error: Failed to connect socket to '/var/run/libvirt/libvirt-sock': No such file or directory

Start the libvirt daemon

On Arch:

```
systemctl start libvirtd
```

On Ubuntu 14.04.1:

```
service libvirt-bin start
```

#### error: Failed to connect socket to '/var/run/libvirt/libvirt-sock': Permission denied

Fix libvirt access permission (Remember to adapt `$USER`)

On Arch and Fedora 21:

```
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

```
usermod -a -G libvirtd $USER
```

#### error: Out of memory initializing network (virsh net-create...)

Ensure libvirtd has been restarted since ebtables was installed.
