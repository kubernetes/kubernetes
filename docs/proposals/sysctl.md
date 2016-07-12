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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

This proposal aims at extending the current pod specification with support
for namespaced kernel parameters set for each pod.

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
  [docker#4717](https://github.com/docker/docker/issues/4717#issuecomment-98653017)).

## Constraints and Assumptions

* Only namespaced kernel parameters can be modified
* Built on-top of the existing security context work
* Be container-runtime agnostic
  - on the API level
  - the implementation (and the set of supported sysctls) will depend on the runtime
* Kernel parameters can be set during a container creation process only.

## Further work (out of scope for this proposal)

* Update kernel parameters in running containers.
* Integration with new container runtime proposal: https://github.com/kubernetes/kubernetes/pull/25899.
* Hugepages support (compare [docker#4717](https://github.com/docker/docker/issues/4717#issuecomment-77426026)) - while also partly configured through sysctls (`vm.nr_hugepages`, compare http://andrigoss.blogspot.de/2008/02/jvm-performance-tuning.html) - is out-of-scope for this proposal as it is not namespaced and as a limited resource (similar to normal memory) needs deeper integration e.g. with the scheduler.

## Use Cases

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

## Community Work

### Docker support for sysctl

Supported sysctls (whitelist) as of Docker 1.12.0:

- IPC namespace
  - System V: `kernel.msgmax`, `kernel.msgmnb`, `kernel.msgmni`, `kernel.sem`,
    `kernel.shmall`, `kernel.shmmax`, `kernel.shmmni`, `kernel.shm_rmid_forced`
  - POSIX queues: `fs.mqueue.*`
- network namespace: `net.*`

Error behaviour:

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

## Proposed Design

Sysctls in pods and `PodSecurityPolicy` are first introduced as an alpha feature for Kubernetes 1.4. This means that the following changes apply to the internal unversioned API only. The externally visible v1 API will model these as annotations, with the plan to turn those in first class citizens in a later release.

### (Internal) pod API changes

Container specification must be changed to allow the specification of kernel parameters:

```go
// Sysctl defines a kernel parameter to be set
type Sysctl struct {
	// Name of a property to set
	Name string `json:"name"`
	// Value of a property to set
	Value string `json:"value"`
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

Note that sysctls must be on the pod level because containers in a pod share IPC and network namespaces (if pod.spec.hostIPC and pod.spec.hostNetwork is false) and therefore cannot have conflicting sysctl values. Moreover, note that all namespaced sysctl supported by Docker/RunC are either in the IPC or network namespace.

### Annotations for the (external) versioned API

- `sysctls.alpha.kubernetes.io/pod-sysctl-1: kernel.shm_rmid_forced:4`
- `sysctls.alpha.kubernetes.io/pod-sysctl-2: kernel.msgmax:foo,bar`
- ...

The same annotation keys are used for pods and `PodSecurityPolicy` objects.

The value is everything behind the colon.

Note that the use of the enumerations for multiple keys is intentional. An alternative would be to encode a list of key/values pairs in one annotation. But as abritrary values are supposed to work, an encoded list is much harder to use. Another alternative would be encode the sysctl name, e.g. `netIpv6Idgen-delay` for `net.ipv6.idgen_delay` (neither dots nor underscore are valid characters in an annotation key). But that does not make it easier either.

### Validation

The name of each sysctl in `PodSecurityContext.Name` is validated against a static list of

- specific sysctls
- and a list of sysctl prefixes,

all known to be namespaced by the kernel. These two lists are defined under `pkg/kubelet` and to be maintained by the nodes team.

The initial whitelists will be:

```go
var whitelist = map[string]struct{}{
       "kernel.msgmax":          {},
       "kernel.msgmnb":          {},
       "kernel.msgmni":          {},
       "kernel.sem":             {},
       "kernel.shmall":          {},
       "kernel.shmmax":          {},
       "kernel.shmmni":          {},
       "kernel.shm_rmid_forced": {},
}

var whitelistPrefixes = []string{
       "net.",
       "fs.mqueue.",
}
```

In addition to the whitelisting, the general format of the sysctl name will be checked:
- 253 characters in length
- it matches `sysctlRegexp`:

```go
const SysctlSegmentFmt string = "[a-z0-9]([_a-z0-9]*[a-z0-9])?"
const SysctlFmt string = "(" + SysctlSegmentFmt + "\\.)*" + SysctlSegmentFmt
var sysctlRegexp = regexp.MustCompile("^" + SysctlFmt + "$")
```

### Error behaviour

Pods with specified sysctls are either launched with the given sysctl values or fail to launch, creating an event which is verbose enough to tell the user what happened.

### Scheduling

For the first step of sysctl support it is considered *out of scope* to schedule pods with certain sysctls onto certain hosts according to some given rules.

One could imagine to offer certain non-namespaced sysctls as well which taint a host such that only containers with compatible sysctls settings are scheduled there. Moreover, there is no accounting of sysctl values if that makes any sense for a given sysctl.

### SecurityContext Enforcement

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

The default policy will be `*`, i.e. all whitelisted (and therefore known-to-be-namespaced) sysctls are allowed.

### Application of the given Sysctls

Finally, the container runtime will interpret `pod.spec.securityPolicy.sysctls`,
e.g. in the case of Docker the `DockerManager` will apply the given sysctls to the infra container in `createPodInfraContainer`.

In a later implementation of a container runtime interface (compare https://github.com/kubernetes/kubernetes/pull/25899), sysctls will be part of `LinuxPodSandboxConfig` (compare https://github.com/kubernetes/kubernetes/pull/25899#discussion_r64867763) and to be applied by the runtime implementaiton to the `PodSandbox` by the `PodSandboxManager` implementation.

## Examples

### Use in a pod

Here is an example of a pod that has `net.ipv4.ip_forward` set to `2`:

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
    - name: net.ipv4.ip_forward
      value: "2"
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
