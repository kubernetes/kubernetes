<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Setting Sysctls on the Pod Level

This proposal aims at extending the current pod specification with support
for namespaced kernel parameters (sysctls) set for each pod.

## Roadmap

### v1.4

- [ ] initial implementation for v1.4 https://github.com/kubernetes/kubernetes/pull/27180
  + node-level whitelist for safe sysctls: `kernel.shm_rmid_forced`, `net.ipv4.ip_local_port_range`, `net.ipv4.tcp_max_syn_backlog`, `net.ipv4.tcp_syncookies`
  + (disabled by-default) unsafe sysctls: `kernel.msg*`, `kernel.sem`, `kernel.shm*`, `fs.mqueue.*`, `net.*`
  + new kubelet flag: `--experimental-allowed-unsafe-sysctls`
  + PSP default: `*`
- [ ] document node-level whitelist with kubectl flags and taints/tolerations
- [ ] document host-level sysctls with daemon sets + taints/tolerations
- in parallel: kernel upstream patches to fix ipc accounting for 4.5+
  + [ ] submitted to mainline
  + [ ] merged into mainline

### v1.5

- pre-requisites for `kernel.sem`, `kernel.msg*`, `fs.mqueue.*` on the node-level whitelist
  + [ ] pod cgroups active by default (compare [Pod Resource Management](pod-resource-management.md#implementation-status))
  + [ ] kmem accounting active by default
  + [ ] kernel patches for 4.5+
- reconsider what to do with `kernel.shm*` and other resource-limit sysctls with proper isolation: (a) keep them in the API (b) set node-level defaults

## Table of Contents

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Setting Sysctls on the Pod Level](#setting-sysctls-on-the-pod-level)
  - [Roadmap](#roadmap)
    - [v1.4](#v14)
    - [v1.5](#v15)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Motivation](#motivation)
  - [Abstract Use Cases](#abstract-use-cases)
  - [Constraints and Assumptions](#constraints-and-assumptions)
  - [Further work (out of scope for this proposal)](#further-work-out-of-scope-for-this-proposal)
  - [Community Work](#community-work)
    - [Docker support for sysctl](#docker-support-for-sysctl)
    - [Runc support for sysctl](#runc-support-for-sysctl)
    - [Rkt support for sysctl](#rkt-support-for-sysctl)
  - [Design Alternatives and Considerations](#design-alternatives-and-considerations)
  - [Analysis of Sysctls of Interest](#analysis-of-sysctls-of-interest)
    - [Summary of Namespacing and Isolation](#summary-of-namespacing-and-isolation)
    - [Classification](#classification)
  - [Proposed Design](#proposed-design)
    - [Pod API Changes](#pod-api-changes)
    - [Apiserver Validation and Kubelet Admission](#apiserver-validation-and-kubelet-admission)
      - [In the Apiserver](#in-the-apiserver)
      - [In the Kubelet](#in-the-kubelet)
    - [Error behavior](#error-behavior)
    - [Kubelet Flags to Extend the Whitelist](#kubelet-flags-to-extend-the-whitelist)
    - [SecurityContext Enforcement](#securitycontext-enforcement)
      - [Alternative 1: by name](#alternative-1-by-name)
      - [Alternative 2: SysctlPolicy](#alternative-2-sysctlpolicy)
    - [Application of the given Sysctls](#application-of-the-given-sysctls)
  - [Examples](#examples)
    - [Use in a pod](#use-in-a-pod)
    - [Allowing only certain sysctls](#allowing-only-certain-sysctls)

<!-- END MUNGE: GENERATED_TOC -->

## Abstract

In Linux, the sysctl interface allows an administrator to modify kernel
parameters at runtime. Parameters are available via `/proc/sys/` virtual
process file system. The parameters cover various subsystems such as:

* kernel (common prefix: `kernel.`)
* networking (common prefix: `net.`)
* virtual memory (common prefix: `vm.`)
* MDADM (common prefix: `dev.`)

More subsystems are described in [Kernel docs](https://www.kernel.org/doc/Documentation/sysctl/README).

To get a list of basic prefixes on your system, you can run

```
$ sudo sysctl -a | cut -d' ' -f1 | cut -d'.' -f1 | sort -u
```

To get a list of all parameters, you can run

```
$ sudo sysctl -a
```

A number of them are namespaced and can therefore be set for a container
independently with today's Linux kernels.

**Note**: This proposal - while sharing some use-cases - does not cover ulimits
(compare [Expose or utilize docker's rlimit support](https://github.com/kubernetes/kubernetes/issues/3595)).

## Motivation

A number of Linux applications need certain kernel parameter settings to

- either run at all
- or perform well.

In Kubernetes we want to allow to set these parameters within a pod specification
in order to enable the use of the platform for those applications.

With Docker version 1.11.1 it is possible to change kernel parameters inside privileged containers.
However, the process is purely manual and the changes might be applied across all containers
affecting the entire host system. It is not possible to set the parameters within a non-privileged
container.

With [docker#19265](https://github.com/docker/docker/pull/19265) docker-run as of 1.12.0
supports setting a number of whitelisted sysctls during the container creation process.

Some real-world examples for the use of sysctls:

- PostgreSQL requires `kernel.shmmax` and `kernel.shmall` (among others) to be
  set to reasonable high values (compare [PostgresSQL Manual 17.4.1. Shared Memory
  and Semaphores](http://www.postgresql.org/docs/9.1/static/kernel-resources.html)).
  The default of 32 MB for shared memory is not reasonable for a database.
- RabbitMQ proposes a number of sysctl settings to optimize networking: https://www.rabbitmq.com/networking.html.
- web applications with many concurrent connections require high values for
  `net.core.somaxconn`.
- a containerized IPv6 routing daemon requires e.g. `/proc/sys/net/ipv6/conf/all/forwarding` and
  `/proc/sys/net/ipv6/conf/all/accept_redirects` (compare
  [docker#4717](https://github.com/docker/docker/issues/4717#issuecomment-98653017))
- the [nginx ingress controller in kubernetes/contrib](https://github.com/kubernetes/contrib/blob/master/ingress/controllers/nginx/examples/sysctl/change-proc-values-rc.yaml#L80)
  uses a privileged sidekick container to set `net.core.somaxconn` and `net.ipv4.ip_local_port_range`.
- a huge software-as-a-service provider uses shared memory (`kernel.shm*`) and message queues (`kernel.msg*`) to
  communicate between containers of their web-serving pods, configuring up to 20 GB of shared memory.

  For optimal network layer performance they set `net.core.rmem_max`, `net.core.wmem_max`,
  `net.ipv4.tcp_rmem` and `net.ipv4.tcp_wmem` to much higher values than kernel defaults.

- In [Linux Tuning guides for 10G ethernet](https://fasterdata.es.net/host-tuning/linux/) it is suggested to
  set `net.core.rmem_max`/`net.core.wmem_max` to values as high as 64 MB and similar dimensions for
  `net.ipv4.tcp_rmem`/`net.ipv4.tcp_wmem`.

  It is noted that
  > tuning settings described here will actually decrease performance of hosts connected at rates of OC3 (155 Mbps) or less.

- For integration of a web-backend with the load-balancer retry mechanics it is suggested in http://serverfault.com/questions/518862/will-increasing-net-core-somaxconn-make-a-difference:

  > Sometimes it's preferable to fail fast and let the load-balancer to do it's job(retry) than to make user wait - for that purpose we set net.core.somaxconn any value, and limit application backlog to e.g. 10 and set net.ipv4.tcp_abort_on_overflow to 1.

  In other words, sysctls change the observable application behavior from the view of the load-balancer radically.

## Abstract Use Cases

As an administrator I want to set customizable kernel parameters for a container

1. To be able to limit consumed kernel resources
   1. so I can provide more resources to other containers
   1. to restrict system communication that slows down the host or other containers
   1. to protect against programming errors like resource leaks
   1. to protect against DDoS attacks.
1. To be able to increase limits for certain applications while not
   changing the default for all containers on a host
   1. to enable resource hungry applications like databases to perform well
      while the default limits for all other applications can be kept low
   1. to enable many network connections e.g. for web backends
   1. to allow special memory management like Java hugepages.
1. To be able to enable kernel features.
   1. to enable containerized execution of special purpose applications without
      the need to enable those kernel features host wide, e.g. ip forwarding for
      network router daemons

## Constraints and Assumptions

* Only namespaced kernel parameters can be modified
* Resource isolation is ensured for all safe sysctls. Sysctl with unclear, weak or not existing isolation are called unsafe sysctls. The later are disabled by default.
* Built on-top of the existing security context work
* Be container-runtime agnostic
  - on the API level
  - the implementation (and the set of supported sysctls) will depend on the runtime
* Kernel parameters can be set during a container creation process only.

## Further work (out of scope for this proposal)

* Update kernel parameters in running containers.
* Integration with new container runtime proposal: https://github.com/kubernetes/kubernetes/pull/25899.
* Hugepages support (compare [docker#4717](https://github.com/docker/docker/issues/4717#issuecomment-77426026)) - while also partly configured through sysctls (`vm.nr_hugepages`, compare http://andrigoss.blogspot.de/2008/02/jvm-performance-tuning.html) - is out-of-scope for this proposal as it is not namespaced and as a limited resource (similar to normal memory) needs deeper integration e.g. with the scheduler.

## Community Work

### Docker support for sysctl

Supported sysctls (whitelist) as of Docker 1.12.0:

- IPC namespace
  - System V: `kernel.msgmax`, `kernel.msgmnb`, `kernel.msgmni`, `kernel.sem`,
    `kernel.shmall`, `kernel.shmmax`, `kernel.shmmni`, `kernel.shm_rmid_forced`
  - POSIX queues: `fs.mqueue.*`
- network namespace: `net.*`

Error behavior:

- not whitelisted sysctls are rejected:

```shell
$ docker run --sysctl=foo=bla -it busybox /bin/sh
invalid value "foo=bla" for flag --sysctl: sysctl 'foo=bla' is not whitelisted
See 'docker run --help'.
```

Applied changes:

* https://github.com/docker/docker/pull/19265
* https://github.com/docker/engine-api/pull/38

Related issues:

* https://github.com/docker/docker/issues/21126
* https://github.com/ibm-messaging/mq-docker/issues/13

### Runc support for sysctl

Supported sysctls (whitelist) as of RunC 0.1.1 (compare
[libcontainer config validator](https://github.com/opencontainers/runc/blob/master/libcontainer/configs/validate/validator.go#L107)):

- IPC namespace
  - System V: `kernel.msgmax`, `kernel.msgmnb`, `kernel.msgmni`, `kernel.sem`,
    `kernel.shmall`, `kernel.shmmax`, `kernel.shmmni`, `kernel.shm_rmid_forced`
  - POSIX queues: `fs.mqueue.*`
- network namespace: `net.*`

Applied changes:

* https://github.com/opencontainers/runc/pull/73
* https://github.com/opencontainers/runc/pull/303
*

### Rkt support for sysctl

The only sysctl support in rkt is through a [CNI plugin](https://github.com/containernetworking/cni/blob/master/Documentation/tuning.md) plugin. The Kubernetes network plugin `kubenet` can easily be extended to call this with a given list of sysctls during pod launch.

The default network plugin for rkt is `no-op` though. This mode leaves all network initialization to rkt itself. Rkt in turn uses the static CNI plugin configuration in `/etc/rkt/net.d`. This does not allow to customize the sysctls for a pod. Hence, in order to implement this proposal in `no-op` mode additional changes in rkt are necessary.

Supported sysctls (whitelist):

- network namespace: `net.*`

Applied changes:

* https://github.com/coreos/rkt/issues/2140

Issues:

* https://github.com/coreos/rkt/issues/2075

## Design Alternatives and Considerations

- Each pod has its own network stack that is shared among its containers.
  A privileged side-kick or init container (compare https://github.com/kubernetes/contrib/blob/master/ingress/controllers/nginx/examples/sysctl/change-proc-values-rc.yaml#L80)
  is able to set `net.*` sysctls.

  Clearly, this is completely uncontrolled by the kubelet, but is a usable work-around if privileged
  containers are permitted in the environment. As privileged container permissions (in the admission controller) are an all-or-nothing
  decision and the actual code executed in them is not limited, allowing privileged container might be a security threat.

  The same work-around also works for shared memory and message queue sysctls as they are shared among the containers of a pod
  in their ipc namespace.

- Instead of giving the user a way to set sysctls for his pods, an alternative seems to be to set high values
  for the limits of interest from the beginning inside the kubelet or the runtime. Then - so the theory - the
  user's pods operate under quasi unlimited bounds.

  This might be true for some of the sysctls, which purely set limits for some host resources, but

  * some sysctls influence the behavior of the application, e.g.:
    * `kernel.shm_rmid_forced` adds a garbage collection semantics to shared memory segments when possessing processes die.
      This is against the System V standard though.
    * `net.ipv4.tcp_abort_on_overflow` makes the kernel send RST packets when the application is overloaded, giving a load-balancer
      the chance to reschedule a request to another backend.
  * some sysctls lead to changed resource requirement characteristics, e.g.:
    * `net.ipv4.tcp_rmem`/`net.ipv4.tcp_wmem` not only define min and max values, but also the default tcp window buffer size
      for each socket. While large values are necessary for certain environments and applications, they lead to waste of resources
      in the 90% case.
  * some sysctls have a different error behavior, e.g.:
    * creating a shared memory segment will fail immediately when `kernel.shmmax` is too small.

      With a large `kernel.shmmax` default, the creation of a segment always succeeds, but the OOM killer will
      do its job when a shared memory segment exceeds the memory request of the container.

  The high values that could be set by the kubelet on launch might depend on the node's capacity and capabilities. But for
  portability of workloads it is helpful to have a common baseline of sysctls settings one can expect on every node. The
  kernel defaults (which are active if the kubelet does not change defaults) are such a (natural) baseline.

- One could imagine to offer certain non-namespaced sysctls as well which
  taint a host such that only containers with compatible sysctls settings are
  scheduled there. This is considered *out of scope* to schedule pods with certain sysctls onto certain hosts according to some given rules. This must be done manually by the admin, e.g. by using taints and tolerations.

- (Next to namespacing) *isolation* is the key requirement for a sysctl to be unconditionally allowed in a pod spec. There are the following alternatives:

  1. allow only namespaced **and** isolated sysctls (= safe) in the API
  2. allow only namespaced **and** isolated sysctls **by-default** and make all other namespaced sysctls with unclear or weak isolation (= unsafe) opt-in by the cluster admin.

  For v1.4 only a handful of *safe* sysctls are defined. There are known, non-edge-case use-cases (see above) for a number of further sysctls. Some of them (especially the ipc sysctls) will probably be promoted onto the whitelist of safe sysctls in the near future when Kubernetes implements better resource isolation.

  On the other hand, especially in the `net.*` hierarchy there are a number of very low-level knobs to tune the network stack. They might be necessary for classes of applications requiring high-performance or realtime behavior. It is hard to forsee which knobs will be necessary in the future. At the same time the `net.*` hierarchy is huge making deep analysis on a 1-on-1 basis hard. If there is no way to use them at-your-own-risk, those users are forced into the use of privileged containers. This might be a security threat and a no-go for certain environments. Sysctls in the API (even if unsafe) in contrast allow finegrained control by the cluster admin without essentially opening up root access to the cluster nodes for some users.

  This requirement for a large number of accessible sysctls must be balanced though with the desire to have a minimal API surface: removing certain (unsafe) sysctls from an official API in a later version (e.g. because they turned out to be problematic for the node health) is problematic.

  To balance those two desires the API can be split in half: one official way to declare *safe* sysctls in a pod spec (this one will be promoted to beta and stable some day) and an alternative way to define *unsafe* sysctls. Possibly the second way will stay alpha forever to make it clear that unsafe sysctls are not a stable API of Kubernetes. Moreover, for all *unsafe* sysctls an opt-in policy is desirable, only controllable by the cluster admin, not by each cluster user.

## Analysis of Sysctls of Interest

**Note:** The kmem accounting has fundamentally changed in kernel 4.5 (compare https://github.com/torvalds/linux/commit/a9bb7e620efdfd29b6d1c238041173e411670996): older kernels (e.g. 4.4 from Ubuntu 16.04, 3.10 from CentOS 7.2) use a blacklist (`__GFP_NOACCOUNT`), newer kernels (e.g. 4.6.x from Fedora 24) use a whitelist (`__GFP_ACCOUNT`). **In the following the analysis is done for kernel >= 4.5:**

- `kernel.shmall`, `kernel.shmmax`, `kernel.shmmni`: configure System V shared memory
  * [x] **namespaced** in ipc ns
  * [x] **accounted for** as user memory in memcg, using sparse allocation (like tmpfs)
    uses [Resizable virtual memory filesystem](https://github.com/torvalds/linux/blob/master/mm/shmem.c)
  * [x] hence **safe to customize**
  * [x] **no application influence** with high values
  * **defaults to** [unlimited pages, unlimited size, 4096 segments on today's kernels](https://github.com/torvalds/linux/blob/0e06f5c0deeef0332a5da2ecb8f1fcf3e024d958/include/uapi/linux/shm.h#L20). This makes **customization practically unneccessary**, at least for the segment sizes. IBM's DB2 suggests `256*GB of RAM` for `kernel.shmmni` (compare http://www.ibm.com/support/knowledgecenter/SSEPGG_10.1.0/com.ibm.db2.luw.qb.server.doc/doc/c0057140.html), exceeding the kernel defaults for machines with >16GB of RAM.
- `kernel.shm_rmid_forced`: enforce removal of shared memory segments on process shutdown
  * [x] **namespaced** in ipc ns
- `kernel.msgmax`, `kernel.msgmnb`, `kernel.msgmni`: configure System V messages
  * [x] **namespaced** in ipc ns
  * [ ] [temporarily **allocated in kmem** in a linked message list](http://lxr.linux.no/linux+v4.7/ipc/msgutil.c#L58), but **not accounted for** in memcg **with kernel >= 4.5**
  * [ ] **defaults to** [8kb max packet size, 16384 kb total queue size, 32000 queues](http://lxr.linux.no/linux+v4.7/include/uapi/linux/msg.h#L75), **which might be too small** for certain applications
  * [ ] arbitrary values [up to INT_MAX](http://lxr.linux.no/linux+v4.7/ipc/ipc_sysctl.c#L135). Hence, **potential DoS attack vector** against the host.

  Even without using a sysctl the kernel default allows any pod to allocate 512 MB of message memory (compare https://github.com/sttts/kmem-ipc-msg-queues as a test-case). If kmem acconting is not active, this is outside of the pod resource limits. Then a node with 8 GB will not survive with >16 replicas of such a pod.

- `fs.mqueue.*`: configure POSIX message queues.
  * [x] **namespaced** in ipc ns
  * [ ] uses the same [`load_msg`](http://lxr.linux.no/linux+v4.7/ipc/msgutil.c#L58) as System V messages, i.e. **no accounting for kernel >= 4.5**
  * does [strict checking against rlimits](http://lxr.free-electrons.com/source/ipc/mqueue.c#L278) though
  * [ ] **defaults to** [256 queues, max queue length 10, message size 8kb](http://lxr.free-electrons.com/source/include/linux/ipc_namespace.h#L102)
  * [ ] can be customized via sysctls up to 64k max queue length, message size 16MB. Hence, **potential DoS attack vector** against the host
- `kernel.sem`: configure System V semaphores
  * [x] **namespaced** in ipc ns
  * [ ] uses [plain kmalloc and vmalloc](http://lxr.free-electrons.com/source/ipc/util.c#L404) **without accounting**
  * [x] **defaults to** [32000 ids and 32000 semaphores per id](http://lxr.free-electrons.com/source/include/uapi/linux/sem.h#L78) (needing double digit number of bytes each), probably enough for all applications:

    > The values has been chosen to be larger than necessary for any known configuration. ([linux/sem.h](http://lxr.free-electrons.com/source/include/uapi/linux/sem.h#L69))

- `net.*`: configure the network stack
  - `net.core.somaxconn`: maximum queue length specifiable by listen.
    * [x] **namespaced** in net ns
    * [ ] **might have application influence** for high values as it limits the socket queue length
    * [?] **No real evidence found until now for accounting**. The limit is checked by `sk_acceptq_is_full` at http://lxr.free-electrons.com/source/net/ipv4/tcp_ipv4.c#L1276. After that a new socket is created. Probably, the tcp socket buffer sysctls apply then, with their accounting, see below.
    * [ ] **very unreliable** tcp memory accounting. There have a been a number of attemps to drop that from the kernel completely, e.g. https://lkml.org/lkml/2014/9/12/401. On Fedora 24 (4.6.3) tcp accounting did not work at all, on Ubuntu 16.06 (4.4) it kind of worked in the root-cg, but in containers only values copied from the root-cg appeared.
e  - `net.ipv4.tcp_wmem`/`net.ipv4.tcp_wmem`/`net.core.rmem_max`/`net.core.wmem_max`: socket buffer sizes
    * [ ] **not namespaced in net ns**, and they are not even available under `/sys/net`
  - `net.ipv4.ip_local_port_range`: local tcp/udp port range
    * [x] **namespaced** in net ns
    * [x] **no memory involved**
  - `net.ipv4.tcp_max_syn_backlog`: number of half-open connections
    * [ ] **not namespaced**
  - `net.ipv4.tcp_syncookies`: enable syn cookies
    * [x] **namespaced** in net ns
    * [x] **no memory involved**

### Summary of Namespacing and Isolation

The individual analysis above leads to the following summary of:

- namespacing (ns) - the sysctl is set in this namespace, independently from the parent/root namespace
- accounting (acc.) - the memory resources caused by the sysctl are accounted for by the given cgroup

  Kernel <= 4.4 and >= 4.5 fundamentally different kernel memory accounting (see note above). The two columns describe the two cases.

| sysctl                       | ns   | acc. for <= 4.4 | >= 4.5        |
| ---------------------------- | ---- | --------------- | ------------- |
| kernel.shm*                  | ipc  | user memcg 1)   | user memcg 1) |
| kernel.msg*                  | ipc  | kmem memcg 3)   | - 3)          |
| fs.mqueue.*                  | ipc  | kmem memcg      | -             |
| kernel.sem                   | ipc  | kmem memcg      | -             |
| net.core.somaxconn           | net  | unreliable 4)   | unreliable 4) |
| net.*.tcp_wmem/rmem          | - 2) | unreliable 4)   | unreliable 4) |
| net.core.wmem/rmem_max       | - 2) | unreliable 4)   | unreliable 4) |
| net.ipv4.ip_local_port_range | net  | not needed 5)   | not needed 5) |
| net.ipv4.tcp_syncookies      | net  | not needed 5)   | not needed 5) |
| net.ipv4.tcp_max_syn_backlog | - 2) | ?               | ?             |

Footnotes:

1. a pod memory cgroup is necessary to catch segments from a dying process.
2. only available in root-ns, not even visible in a container
3. compare https://github.com/sttts/kmem-ipc-msg-queues as a test-case
4. in theory socket buffers should be accounted for by the kmem.tcp memcg counters. In practice this only worked very unreliably and not reproducibly, on some kernel not at all. kmem.tcp acconuting seems to be deprecated and on lkml patches has been posted to drop this broken feature.
5. b/c no memory is involved, i.e. purely functional difference

**Note**: for all sysctls marked as "kmem memcg" kernel memory accounting must be enabled in the container for proper isolation. This will not be the case for 1.4, but is planned for 1.5.

### Classification

From the previous analysis the following classification is derived:

| sysctl                       | ns    | accounting | reclaim   | pre-requisites |
| ---------------------------- | ----- | ---------- | --------- | -------------- |
| kernel.shm*                  | pod   | container  | pod       | i 1)           |
| kernel.msg*                  | pod   | container  | pod       | i + ii + iii   |
| fs.mqueue.*                  | pod   | container  | pod       | i + ii + iii   |
| kernel.sem                   | pod   | container  | pod       | i + ii + iii   |
| net.core.somaxconn           | pod   | container  | container | i + ii + iv    |
| net.*.tcp_wmem/rmem          | host  | container  | container | i + ii + iv    |
| net.core.wmem/rmem_max       | host  | container  | container | i + ii + iv    |
| net.ipv4.ip_local_port_range | pod   | n/a        | n/a       | -              |
| net.ipv4.tcp_syncookies      | pod   | n/a        | n/a       | -              |
| net.ipv4.tcp_max_syn_backlog | pod   | n/a        | n/a       | -              |

Explanation:

- ns: value is namespaced on this level
- accounting: memory is accounted for against limits of this level
- reclaim: in the worst case, memory resources fall-through to this level and are accounted for there until they get destroyed
- pre-requisites:
  1. pod level cgroups
  2. kmem acconuting enabled in Kubernetes
  3. kmem accounting fixes for ipc namespace in Kernel >= 4.5
  4. reliable kernel tcp net buffer accounting, which probably means to wait for cgroups v2.

Footnote:

1. Pod level cgroups don't exist today and pages are already re-parented on container deletion in v1.3. So supporting pod level sysctls in v1.4 that are tracked by user space memcg is not introducing any regression.

**Note**: with the exception of `kernel.shm*` all of the listed pod-level sysctls depend on kernel memory accounting to be enabled for proper resource isolation. This will not be the case for 1.4 by default, but is planned in 1.5.

**Note**: all the ipc objects persist when the originating containers dies. Their resources (if kmem accounting is enabled) fall back to the parent cgroup. As long as there is no pod level memory cgroup, the parent will be the container runtime, e.g. the docker daemon or the RunC process. It is [planned with v1.5 to introduce a pod level memory cgroup](pod-resource-management.md#implementation-status) which will fix this problem.

**Note**: in general it is good practice to reserve special nodes for those pods which set sysctls which the kernel does not guarantee proper isolation for.

## Proposed Design

Sysctls in pods and `PodSecurityPolicy` are first introduced as an alpha feature for Kubernetes 1.4. This means that the API will model these as annotations, with the plan to turn those in first class citizens in a later release when the feature is promoted to beta.

It is proposed to use a syntactical validation in the apiserver **and** a node-level whitelist of *safe sysctls* in the kubelet. The whitelist shall be fixed per version and might grow in the future when better resource isolation is in place in the kubelet. In addition a list of *allowed unsafe sysctls* will be configured per node by the cluster admin, with an empty list as the default.

The following rules apply:

- Only sysctls shall be whitelisted in the kubelet
  + that are properly namespaced by the container or the pod (e.g. in the ipc or net namespace)
  + **and** that cannot lead to resource consumption outside of the limits of the container or the pod.
  These are called *safe*.
- The cluster admin shall only be able to manually enable sysctls in the kubelet
  + that are properly namespaced by the container or the pod (e.g. in the ipc or net namespace).
  These are call *unsafe*.

This means that sysctls that are not namespaced must be set by the admin on host level at his own risk, e.g. by running a *privileged daemonset*, possibly limited to a restricted, special-purpose set of nodes, if necessary with the host network namespace. This is considered out-of-scope of this proposal and out-of-scope of what the kubelet will do for the admin. A section is going to be added to the documentation describing this.

The *allowed unsafe sysctls* will be configurable on the node via a flag of the kubelet.

### Pod API Changes

Pod specification must be changed to allow the specification of kernel parameters:

```go
// Sysctl defines a kernel parameter to be set
type Sysctl struct {
	// Name of a property to set
	Name string `json:"name"`
	// Value of a property to set
	Value intstr.IntOrString `json:"value"`
	// Must be true for unsafe sysctls.
	Unsafe bool `json:"unsafe,omitempty"
}

// PodSecurityContext holds pod-level security attributes and common container settings.
// Some fields are also present in container.securityContext.  Field values of
// container.securityContext take precedence over field values of PodSecurityContext.
type PodSecurityContext struct {
	...
	// Sysctls hold a list of namespaced sysctls used for the pod. Pods with unsupported
	// sysctls (by the container runtime) might fail to launch.
	Sysctls []Sysctl `json:"sysctls,omitempty"`
}
```

During alpha the extension of `PodSecurityContext` is modeled with annotations:

```
security.alpha.kubernetes.io/sysctls: kernel.shm_rmid_forced=1`
security.alpha.kubernetes.io/unsafe-sysctls: net.ipv4.route.min_pmtu=1000,kernel.msgmax=1 2 3`
```

The value is a comma separated list of key-value pairs separated by `=`.

*Safe* sysctls may be declared with `unsafe: true` (or in the respective annotation), while for *unsafe* sysctls `unsafe: true` is mandatory. This guarantees backwards-compatibility in future versions when sysctls have been promoted to the whitelist: old pod specs will still work.

Possibly, the `security.alpha.kubernetes.io/unsafe-sysctls` annotation will stay as an alpha API (replacing the `Unsafe bool` field) even when `security.alpha.kubernetes.io/sysctls` has been promoted to beta or stable. This helps to make clear that unsafe sysctls are not a stable feature.

**Note**: none of the whitelisted (and in general none with the exceptions of descriptive plain text ones) sysctls use anything else than numbers, possibly separated with spaces.

**Note**: sysctls must be on the pod level because containers in a pod share IPC and network namespaces (if pod.spec.hostIPC and pod.spec.hostNetwork is false) and therefore cannot have conflicting sysctl values. Moreover, note that all namespaced sysctl supported by Docker/RunC are either in the IPC or network namespace.

### Apiserver Validation and Kubelet Admission

#### In the Apiserver

The name of each sysctl in `PodSecurityContext.Sysctls[*].Name` (or the `annotation security.alpha.kubernetes.io/[unsafe-]sysctls` during alpha) is validated by the apiserver against:

- 253 characters in length
- it matches `sysctlRegexp`:

```go
const SysctlSegmentFmt string = "[a-z0-9]([-_a-z0-9]*[a-z0-9])?"
const SysctlFmt string = "(" + SysctlSegmentFmt + "\\.)*" + SysctlSegmentFmt
var sysctlRegexp = regexp.MustCompile("^" + SysctlFmt + "$")
```

#### In the Kubelet

The name of each sysctl in `PodSecurityContext.Sysctls[*].Name` (or the `annotation security.alpha.kubernetes.io/[unsafe-]sysctls` during alpha) is checked by the kubelet against a static *whitelist*.

The whitelist is defined under `pkg/kubelet` and to be maintained by the nodes team.

The initial whitelist of safe sysctls will be:

```go
var whitelist = []string{
    "kernel.shm_rmid_forced",
    "net.ipv4.ip_local_port_range",
    "net.ipv4.tcp_syncookies",
    "net.ipv4.tcp_max_syn_backlog",
}
```

In parallel a namespace list is maintained with all sysctls and their respective, known kernel namespaces. This is initially derived from Docker's internal sysctl whitelist:

```go
var namespaces = map[string]string{
    "kernel.sem": "ipc",
}

var prefixNamespaces = map[string]string{
    "kernel.msg": "ipc",
    "kenrel.shm": "ipc",
    "fs.mqueue.": "ipc",
    "net.":       "net",
}
```

If a pod is created with host ipc or host network namespace, the respective sysctls are forbidden.

### Error behavior

Pods that do not comply with the syntactical sysctl format will be rejected by the apiserver. Pods that do not comply with the whitelist (or are not manually enabled as *allowed unsafe sysctls* for a node by the cluster admin) will fail to launch. An event will be created by the kubelet to notify the user.

### Kubelet Flags to Extend the Whitelist

The kubelet will get a new flag:

```
--experimental-allowed-unsafe-sysctls  Comma-separated whitelist of unsafe
                                       sysctls or unsafe sysctl patterns
                                       (ending in *). Use these at your own
                                       risk.
```

It defaults to the empty list.

During kubelet launch the given value is checked against the list of known namespaces for sysctls or sysctl prefixes. If a namespace is not known, the kubelet will terminate with an error.

### SecurityContext Enforcement

#### Alternative 1: by name

A list of permissible sysctls is to be added to `pkg/apis/extensions/types.go` (compare [security-context-constraints](security-context-constraints.md)):

```go
// PodSecurityPolicySpec defines the policy enforced.
type PodSecurityPolicySpec struct {
	...
	// Sysctls is a white list of allowed sysctls in a pod spec. Each entry
	// is either a plain sysctl name or ends in "*" in which case it is considered
	// as a prefix of allowed sysctls.
	Sysctls []string `json:"sysctls,omitempty"`
}
```

The `simpleProvider` in `pkg.security.podsecuritypolicy` will validate the value of `PodSecurityPolicySpec.Sysctls` with the sysctls of a given pod in `ValidatePodSecurityContext`.

The default policy will be `*`, i.e. all syntactly correct sysctls are admitted by the `PodSecurityPolicySpec`.

The `PodSecurityPolicySpec` applies to safe and unsafe sysctls in the same way.

During alpha the following annotation will be used:

```
security.alpha.kubernetes.io/sysctls: kernel.shmmax,kernel.msgmax,fs.mqueue.*`
```

on `PodSecurityPolicy` objects to customize the allowed sysctls.

**Note**: This does not override the whitelist or the *allowed unsafe sysctls* on the nodes. They still apply. This only changes admission of pods in the apiserver. Pods can still fail to launch due to failed admission on the kubelet.

#### Alternative 2: SysctlPolicy

```go
// SysctlPolicy defines how a sysctl may be set. If neither Values,
// nor Min, Max are set, any value is allowed.
type SysctlPolicy struct {
    // Name is the name of a sysctl or a pattern for a name. It consists of
    // dot separated name segments. A name segment matches [a-z]+[a-z_-0-9]* or
    // equals "*". The later is interpretated as a wildcard for that name
    // segment.
    Name string `json:"name"`

    // Values are allowed values to be set. Either Values is
    // set or Min and Max.
    Values []string `json:"values,omitempty"`

    // Min is the minimal value allowed to be set.
    Min *int64 `json:"min,omitempty"`

    // Max is the maximum value allowed to be set.
    Max *int64 `json:"max,omitempty"`
}

// PodSecurityPolicySpec defines the policy enforced on sysctls.
type PodSecurityPolicySpec struct {
    ...
    // Sysctls is a white list of allowed sysctls in a pod spec.
    Sysctls []SysctlPolicy `json:"sysctls,omitempty"`
}
```

During alpha the following annotation will be used:

```
security.alpha.kubernetes.io/sysctls: kernel.shmmax,kernel.msgmax=max:10:min:1,kernel.msgmni=values:1000 2000 3000`
```

This extended syntax is a natural extension of that of alternative 1 and therefore can be implemented any time during alpha.

Alternative 1 or 2 has to be chosen for the external API once the feature is promoted to beta.

### Application of the given Sysctls

Finally, the container runtime will interpret `pod.spec.securityPolicy.sysctls`,
e.g. in the case of Docker the `DockerManager` will apply the given sysctls to the infra container in `createPodInfraContainer`.

In a later implementation of a container runtime interface (compare https://github.com/kubernetes/kubernetes/pull/25899), sysctls will be part of `LinuxPodSandboxConfig` (compare https://github.com/kubernetes/kubernetes/pull/25899#discussion_r64867763) and to be applied by the runtime implementaiton to the `PodSandbox` by the `PodSandboxManager` implementation.

## Examples

### Use in a pod

Here is an example of a pod that has the safe sysctl `net.ipv4.ip_local_port_range` set to `1024 65535` and the unsafe sysctl `net.ipv4.route.min_pmtu` to `1000`.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  labels:
    name: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
  securityContext:
    sysctls:
    - name: net.ipv4.ip_local_port_range
      value: "1024 65535"
    - name: net.ipv4.route.min_pmtu
      value: 1000
      unsafe: true
```

### Allowing only certain sysctls

Here is an example of a `PodSecurityPolicy`, allowing `kernel.shmmax`, `kernel.shmall` and all `net.*`
sysctls to be set:

```yaml
apiVersion: v1
kind: PodSecurityPolicy
metadata:
  name: database
spec:
  sysctls:
  - kernel.shmmax
  - kernel.shmall
  - net.*
```

and a restricted default `PodSecurityPolicy`:

```yaml
apiVersion: v1
kind: PodSecurityPolicy
metadata:
  name:
spec:
  sysctls: # none
```

in contrast to a permissive default `PodSecurityPolicy`:

```yaml
apiVersion: v1
kind: PodSecurityPolicy
metadata:
  name:
spec:
  sysctls:
  - *
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/sysctl.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
