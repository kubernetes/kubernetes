# Running rkt with KVM stage1

rkt has support for executing pods with KVM hypervisor - [LKVM][lkvm] or [QEMU][qemu] as a [stage1][rkt-arch-stage1]. rkt employs this [alternative stage1][stage1-implementers-guide] to run a pod within a virtual machine with its own operating system kernel and hypervisor isolation, rather than creating a container using Linux cgroups and namespaces.

The KVM stage1 does not yet implement all of the default stage1's features and semantics. While the same app container can be executed under isolation by either stage1, it may require different configuration, especially for networking. However, several deployments of the KVM stage1 are operational outside of CoreOS, and we encourage testing of this feature and welcome your contributions.

## Getting started

For LKVM you can use `stage1-kvm.aci` or `stage1-kvm-lkvm.aci`, for QEMU - `stage1-kvm-qemu.aci` from the official release. You can also build rkt yourself with the right options:

```
$ ./autogen.sh && ./configure --with-stage1-flavors=kvm --with-stage1-kvm-hypervisors=lkvm,qemu && make
```

For more details about configure parameters, see [configure script parameters documentation][build-configure].
This will build the rkt binary and the KVM stage1 aci image in `build-rkt-1.25.0+git/target/bin/`. Depending on the configuration options, it will be `stage1-kvm.aci` (if one hypervisor is set), or `stage1-kvm-lkvm.aci` and `stage1-kvm-qemu.aci` (if you want to have both images built once).

Provided you have hardware virtualization support and the [kernel KVM module][kvm-module] loaded (refer to your distribution for instructions), you can then run an image like you would normally do with rkt:

```
# rkt trust --prefix coreos.com/etcd
prefix: "coreos.com/etcd"
key: "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg"
gpg key fingerprint is: 8B86 DE38 890D DB72 9186  7B02 5210 BD88 8818 2190
	CoreOS ACI Builder <release@coreos.com>
Trusting "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg" for prefix "coreos.com/etcd" without fingerprint review.
Added key for prefix "coreos.com/etcd" at "/etc/rkt/trustedkeys/prefix.d/coreos.com/etcd/8b86de38890ddb7291867b025210bd8888182190"
# rkt run coreos.com/etcd:v2.0.9
rkt: searching for app image coreos.com/etcd:v2.0.9
prefix: "coreos.com/etcd"
key: "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg"
gpg key fingerprint is: 8B86 DE38 890D DB72 9186  7B02 5210 BD88 8818 2190
	CoreOS ACI Builder <release@coreos.com>
Key "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg" already in the keystore
Downloading signature from https://github.com/coreos/etcd/releases/download/v2.0.9/etcd-v2.0.9-linux-amd64.aci.asc
Downloading signature: [=======================================] 819 B/819 B
Downloading ACI: [=============================================] 3.79 MB/3.79 MB
rkt: signature verified:
  CoreOS ACI Builder <release@coreos.com>
[188748.162493] etcd[4]: 2015/08/18 14:10:08 etcd: no data-dir provided, using default data-dir ./default.etcd
[188748.163213] etcd[4]: 2015/08/18 14:10:08 etcd: listening for peers on http://localhost:2380
[188748.163674] etcd[4]: 2015/08/18 14:10:08 etcd: listening for peers on http://localhost:7001
[188748.164143] etcd[4]: 2015/08/18 14:10:08 etcd: listening for client requests on http://localhost:2379
[188748.164603] etcd[4]: 2015/08/18 14:10:08 etcd: listening for client requests on http://localhost:4001
[188748.165044] etcd[4]: 2015/08/18 14:10:08 etcdserver: datadir is valid for the 2.0.1 format
[188748.165541] etcd[4]: 2015/08/18 14:10:08 etcdserver: name = default
[188748.166236] etcd[4]: 2015/08/18 14:10:08 etcdserver: data dir = default.etcd
[188748.166719] etcd[4]: 2015/08/18 14:10:08 etcdserver: member dir = default.etcd/member
[188748.167251] etcd[4]: 2015/08/18 14:10:08 etcdserver: heartbeat = 100ms
[188748.167685] etcd[4]: 2015/08/18 14:10:08 etcdserver: election = 1000ms
[188748.168322] etcd[4]: 2015/08/18 14:10:08 etcdserver: snapshot count = 10000
[188748.168787] etcd[4]: 2015/08/18 14:10:08 etcdserver: advertise client URLs = http://localhost:2379,http://localhost:4001
[188748.169342] etcd[4]: 2015/08/18 14:10:08 etcdserver: initial advertise peer URLs = http://localhost:2380,http://localhost:7001
[188748.169862] etcd[4]: 2015/08/18 14:10:08 etcdserver: initial cluster = default=http://localhost:2380,default=http://localhost:7001
[188748.188550] etcd[4]: 2015/08/18 14:10:08 etcdserver: start member ce2a822cea30bfca in cluster 7e27652122e8b2ae
[188748.190202] etcd[4]: 2015/08/18 14:10:08 raft: ce2a822cea30bfca became follower at term 0
[188748.190381] etcd[4]: 2015/08/18 14:10:08 raft: newRaft ce2a822cea30bfca [peers: [], term: 0, commit: 0, applied: 0, lastindex: 0, lastterm: 0]
[188748.190499] etcd[4]: 2015/08/18 14:10:08 raft: ce2a822cea30bfca became follower at term 1
[188748.206523] etcd[4]: 2015/08/18 14:10:08 etcdserver: added local member ce2a822cea30bfca [http://localhost:2380 http://localhost:7001] to cluster 7e27652122e8b2ae
[188749.489184] etcd[4]: 2015/08/18 14:10:10 raft: ce2a822cea30bfca is starting a new election at term 1
[188749.489460] etcd[4]: 2015/08/18 14:10:10 raft: ce2a822cea30bfca became candidate at term 2
[188749.489583] etcd[4]: 2015/08/18 14:10:10 raft: ce2a822cea30bfca received vote from ce2a822cea30bfca at term 2
[188749.489675] etcd[4]: 2015/08/18 14:10:10 raft: ce2a822cea30bfca became leader at term 2
[188749.489760] etcd[4]: 2015/08/18 14:10:10 raft.node: ce2a822cea30bfca elected leader ce2a822cea30bfca at term 2
[188749.523133] etcd[4]: 2015/08/18 14:10:10 etcdserver: published {Name:default ClientURLs:[http://localhost:2379 http://localhost:4001]} to cluster 7e27652122e8b2ae
```

This output is the same you'll get if you run a container-based rkt.
If you want to see the kernel and boot messages, run rkt with the `--debug` flag.

You can exit pressing `<Ctrl-a x>`.

#### CPU usage
By default, processes will start working on all CPUs if at least one app does not have specfied CPUs.
In the other case, container will be working on aggregate amount of CPUs.

#### Memory
Currently, the memory allocated to the virtual machine is a sum of memory required by each app in pod and additional 128MB required by system. If memory of some app is not specified, app memory will be set on default value (128MB).

### Selecting stage1 at runtime

If you want to run software that requires hypervisor isolation along with trusted software that only needs container isolation, you can [choose which stage1 to use at runtime][rkt-run].

For example, to use the official kvm stage1:

```
# rkt run --stage1-name=coreos.com/rkt/stage1-kvm:1.25.0 coreos.com/etcd:v2.0.9
...
```

If the image is not in the store, `--stage1-name` will perform discovery and fetch the image.

## How does it work?

It leverages the work done by Intel with their [Clear Containers system][clear-containers].
Stage1 contains a Linux kernel that is executed under hypervisor (LKVM or QEMU).
This kernel will then start systemd, which in turn will start the applications in the pod.

A KVM-based rkt is very similar to a container-based one, it just uses hypervisor to execute pods instead of systemd-nspawn.

Here's a comparison of the components involved between a container-based and a KVM based rkt.

Container-based:

```
host OS
  └─ rkt
    └─ systemd-nspawn
      └─ systemd
        └─ chroot
          └─ user-app1
```


KVM based:

```
host OS
  └─ rkt
    └─ hypervisor
      └─ kernel
        └─ systemd
          └─ chroot
            └─ user-app1
```


[build-configure]: build-configure.md
[clear-containers]: https://lwn.net/Articles/644675/
[kvm-module]: http://www.linux-kvm.org/page/Getting_the_kvm_kernel_modules
[lkvm]: https://kernel.googlesource.com/pub/scm/linux/kernel/git/will/kvmtool/+/master/README
[qemu]: http://qemu-project.org/Main_Page
[rkt-arch-stage1]: devel/architecture.md#stage-1
[rkt-run]: subcommands/run.md#use-a-custom-stage-1
[stage1-implementers-guide]: devel/stage1-implementors-guide.md
