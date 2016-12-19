# Kubernetes for multiple platforms

**Author**: Lucas Käldström ([@luxas](https://github.com/luxas))

**Status** (25th of August 2016): Some parts are already implemented; but still there quite a lot of work to be done.

## Abstract

We obviously want Kubernetes to run on as many platforms as possible, in order to make Kubernetes a even more powerful system.
This is a proposal that explains what should be done in order to achieve a true cross-platform container management system.

Kubernetes is written in Go, and Go code is portable across platforms.
Docker and rkt are also written in Go, and it's already possible to use them on various platforms.
When it's possible to run containers on a specific architecture, people also want to use Kubernetes to manage the containers.

In this proposal, a `platform` is defined as `operating system/architecture` or `${GOOS}/${GOARCH}` in Go terms.

The following platforms are proposed to be built for in a Kubernetes release:
 - linux/amd64
 - linux/arm (GOARM=6 initially, but we probably have to bump this to GOARM=7 due to that the most of other ARM things are ARMv7)
 - linux/arm64
 - linux/ppc64le

If there's interest in running Kubernetes on `linux/s390x` too, it won't require many changes to the source now when we've laid the ground for a multi-platform Kubernetes already.

There is also work going on with porting Kubernetes to Windows (`windows/amd64`). See [this issue](https://github.com/kubernetes/kubernetes/issues/22623) for more details.

But note that when porting to a new OS like windows, a lot of os-specific changes have to be implemented before cross-compiling, releasing and other concerns this document describes may apply.

## Motivation

Then the question probably is: Why?

In fact, making it possible to run Kubernetes on other platforms will enable people to create customized and highly-optimized solutions that exactly fits their hardware needs.

Example: [Paypal validates arm64 for real-time data analysis](http://www.datacenterdynamics.com/content-tracks/servers-storage/paypal-successfully-tests-arm-based-servers/93835.fullarticle)

Also, by including other platforms to the Kubernetes party a healthy competition between platforms can/will take place.

Every platform obviously has both pros and cons. By adding the option to make clusters of mixed platforms, the end user may take advantage of the good sides of every platform.

## Use Cases

For a large enterprise where computing power is the king, one may imagine the following combinations:
 - `linux/amd64`: For running most of the general-purpose computing tasks, cluster addons, etc.
 - `linux/ppc64le`: For running highly-optimized software; especially massive compute tasks
 - `windows/amd64`: For running services that are only compatible on windows; e.g. business applications written in C# .NET

For a mid-sized business where efficiency is most important, these could be combinations:
 - `linux/amd64`: For running most of the general-purpose computing tasks, plus tasks that require very high single-core performance.
 - `linux/arm64`: For running webservices and high-density tasks => the cluster could autoscale in a way that `linux/amd64` machines could hibernate at night in order to minimize power usage.

For a small business or university, arm is often sufficient:
 - `linux/arm`: Draws very little power, and can run web sites and app backends efficiently on Scaleway for example.

And last but not least; Raspberry Pi's should be used for [education at universities](http://kubecloud.io/) and are great for **demoing Kubernetes' features at conferences.**

## Main proposal

### Release binaries for all platforms

First and foremost, binaries have to be released for all platforms.
This affects the build-release tools. Fortunately, this is quite straightforward to implement, once you understand how Go cross-compilation works.

Since Kubernetes' release and build jobs run on `linux/amd64`, binaries have to be cross-compiled and Docker images should be cross-built.
Builds should be run in a Docker container in order to get reproducible builds; and `gcc` should be installed for all platforms inside that image (`kube-cross`)

All released binaries should be uploaded to `https://storage.googleapis.com/kubernetes-release/release/${version}/bin/${os}/${arch}/${binary}`

This is a fairly long topic. If you're interested how to cross-compile, see [details about cross-compilation](#cross-compilation-details)

### Support all platforms in a "run everywhere" deployment

The easiest way of running Kubernetes on another architecture at the time of writing is probably by using the docker-multinode deployment. Of course, you may choose whatever deployment you want, the binaries are easily downloadable from the URL above.

[docker-multinode](https://github.com/kubernetes/kube-deploy/tree/master/docker-multinode) is intended to be a "kick-the-tires" multi-platform solution with Docker as the only real dependency (but it's not production ready)

But when we (`sig-cluster-lifecycle`) have standardized the deployments to about three and made them production ready; at least one deployment should support **all platforms**.

### Set up a build and e2e CI's

#### Build CI

Kubernetes should always enforce that all binaries are compiling.
**On every PR, `make release` have to be run** in order to require the code proposed to be merged to be compatible for all architectures.

For more information, see [conflicts](#conflicts)

#### e2e CI

To ensure all functionality really is working on all other platforms, the community should be able to setup a CI.
To be able to do that, all the test-specific images have to be ported to multiple architectures, and the test images should preferably be manifest lists.
If the test images aren't manifest lists, the test code should automatically choose the right image based on the image naming.

IBM volunteered to run continuously running e2e tests for `linux/ppc64le`.
Still it's hard to set up a such CI (even on `linux/amd64`), but that work belongs to `kubernetes/test-infra` proposals.

When it's possible to test Kubernetes using Kubernetes; volunteers should be given access to publish their results on `k8s-testgrid.appspot.com`.

### Official support level

When all e2e tests are passing for a given platform; the platform should be officially supported by the Kubernetes team.
At the time of writing, `amd64` is in the officially supported category category.

When a platform is building and it's possible to set up a cluster with the core functionality, the platform is supported on a "best-effort" and experimental basis.
At the time of writing, `arm`, `arm64` and `ppc64le` are in the experimental category; the e2e tests aren't cross-platform yet.

### Docker image naming and manifest lists

#### Docker manifest lists

Here's a good article about how the "manifest list" in the Docker image [manifest spec v2](https://github.com/docker/distribution/pull/1068) works: [A step towards multi-platform Docker images](https://integratedcode.us/2016/04/22/a-step-towards-multi-platform-docker-images/)

A short summary: A manifest list is a list of Docker images with a single name (e.g. `busybox`), that holds layers for multiple platforms _when it's stored in a registry_.
When the image is pulled by a client (`docker pull busybox`), only layers for the target platforms are downloaded.
Right now we have to write `busybox-${ARCH}` for example instead, but that leads to extra scripting and unnecessary logic.

For reference see [docker/docker#24739](https://github.com/docker/docker/issues/24739) and [appc/docker2aci#193](https://github.com/appc/docker2aci/issues/193)

#### Image naming

This has been debated quite a lot about; how we should name non-amd64 docker images that are pushed to `gcr.io`. See [#23059](https://github.com/kubernetes/kubernetes/pull/23059) and [#23009](https://github.com/kubernetes/kubernetes/pull/23009).

This means that the naming `gcr.io/google_containers/${binary}:${version}` should contain a _manifest list_ for future tags.
The manifest list thereby becomes a wrapper that is pointing to the `-${arch}` images.
This requires `docker-1.10` or newer, which probably means Kubernetes v1.4 and higher.

TL;DR;
 - `${binary}-${arch}:${version}` images should be pushed for all platforms
 - `${binary}:${version}` images should point to the `-${arch}`-specific ones, and docker will then download the right image.

### Components should expose their platform

It should be possible to run clusters with mixed platforms smoothly. After all, bringing heterogenous machines together to a single unit (a cluster) is one of Kubernetes' greatest strengths. And since the Kubernetes' components communicate over HTTP, two binaries of different architectures may talk to each other normally.

The crucial thing here is that the components that handle platform-specific tasks (e.g. kubelet) should expose their platform. In the kubelet case, we've initially solved it by exposing the labels `beta.kubernetes.io/{os,arch}` on every node. This way an user may run binaries for different platforms on a multi-platform cluster, but still it requires manual work to apply the label to every manifest.

Also, [the apiserver now exposes](https://github.com/kubernetes/kubernetes/pull/19905) it's platform at `GET /version`. But note that the value exposed at `/version` only is the apiserver's platform; there might be kubelets of various other platforms.

### Standardize all image Makefiles to follow the same pattern

All Makefiles should push for all platforms when doing `make push`, and build for all platforms when doing `make build`.
Under the hood; they should compile binaries in a container for reproducability, and use QEMU for emulating Dockerfile `RUN` commands if necessary.

### Remove linux/amd64 hard-codings from the codebase

All places where `linux/amd64` is hardcoded in the codebase should be rewritten.

#### Make kubelet automatically use the right pause image

The `pause` is used for connecting containers into Pods. It's a binary that just sleeps forever.
When Kubernetes starts up a Pod, it first starts a `pause` container, and let's all "real" containers join the same network by setting `--net=${pause_container_id}`.

So in order to start Kubernetes Pods on any other architecture, an ever-sleeping image have to exist.

Fortunately, `kubelet` has the `--pod-infra-container-image` option, and it has been used when running Kubernetes on other platforms.

But relying on the deployment setup to specify the right image for the platform isn't great, the kubelet should be smarter than that.

This specific problem has been fixed in [#23059](https://github.com/kubernetes/kubernetes/pull/23059).

#### Vendored packages

Here are two common problems that a vendored package might have when trying to add/update it:
 - Including constants combined with build tags

```go
//+ build linux,amd64
const AnAmd64OnlyConstant = 123
```

 - Relying on platform-specific syscalls (e.g. `syscall.Dup2`)

If someone tries to add a dependency that doesn't satisfy these requirements; the CI will catch it and block the PR until the author has updated the vendored repo and fixed the problem.

### kubectl should be released for all platforms that are relevant

kubectl is released for more platforms than the proposed server platforms, if you want to check out an up-to-date list of them, [see here](../../hack/lib/golang.sh).

kubectl is trivial to cross-compile, so if there's interest in adding a new platform for it, it may be as easy as appending the platform to the list linked above.

### Addons

Addons like dns, heapster and ingress play a big role in a working Kubernetes cluster, and we should aim to be able to deploy these addons on multiple platforms too.

`kube-dns`, `dashboard` and `addon-manager` are the most important images, and they are already ported for multiple platforms.

These addons should also be converted to multiple platforms:
 - heapster, influxdb + grafana
 - nginx-ingress
 - elasticsearch, fluentd + kibana
 - registry

### Conflicts

What should we do if there's a conflict between keeping e.g. `linux/ppc64le` builds vs. merging a release blocker?

In fact, we faced this problem while this proposal was being written; in [#25243](https://github.com/kubernetes/kubernetes/pull/25243). It is quite obvious that the release blocker is of higher priority.

However, before temporarily [deactivating builds](https://github.com/kubernetes/kubernetes/commit/2c9b83f291e3e506acc3c08cd10652c255f86f79), the author of the breaking PR should first try to fix the problem. If it turns out being really hard to solve, builds for the affected platform may be deactivated and a P1 issue should be made to activate them again.

## Cross-compilation details (for reference)

### Go language details

Go 1.5 introduced many changes. To name a few that are relevant to Kubernetes:
 - C was eliminated from the tree (it was earlier used for the bootstrap runtime).
 - All processors are used by default, which means we should be able to remove [lines like this one](https://github.com/kubernetes/kubernetes/blob/v1.2.0/cmd/kubelet/kubelet.go#L37)
 - The garbage collector became more efficent (but also [confused our latency test](https://github.com/golang/go/issues/14396)).
 - `linux/arm64` and `linux/ppc64le` were added as new ports.
 - The `GO15VENDOREXPERIMENT` was started. We switched from `Godeps/_workspace` to the native `vendor/` in [this PR](https://github.com/kubernetes/kubernetes/pull/24242).
 - It's not required to pre-build the whole standard library `std` when cross-compliling. [Details](#prebuilding-the-standard-library-std)
 - Builds are approximately twice as slow as earlier. That affects the CI. [Details](#releasing)
 - The native Go DNS resolver will suffice in the most situations. This makes static linking much easier.

All release notes for Go 1.5 [are here](https://golang.org/doc/go1.5)

Go 1.6 didn't introduce as many changes as Go 1.5 did, but here are some of note:
 - It should perform a little bit better than Go 1.5.
 - `linux/mips64` and `linux/mips64le` were added as new ports.
 - Go < 1.6.2 for `ppc64le` had [bugs in it](https://github.com/kubernetes/kubernetes/issues/24922).

All release notes for Go 1.6 [are here](https://golang.org/doc/go1.6)

In Kubernetes 1.2, the only supported Go version was `1.4.2`, so `linux/arm` was the only possible extra architecture: [#19769](https://github.com/kubernetes/kubernetes/pull/19769).
In Kubernetes 1.3, [we upgraded to Go 1.6](https://github.com/kubernetes/kubernetes/pull/22149), which made it possible to build Kubernetes for even more architectures [#23931](https://github.com/kubernetes/kubernetes/pull/23931).

#### The `sync/atomic` bug on 32-bit platforms

From https://golang.org/pkg/sync/atomic/#pkg-note-BUG:
> On both ARM and x86-32, it is the caller's responsibility to arrange for 64-bit alignment of 64-bit words accessed atomically. The first word in a global variable or in an allocated struct or slice can be relied upon to be 64-bit aligned.

`etcd` have had [issues](https://github.com/coreos/etcd/issues/2308) with this. See [how to fix it here](https://github.com/coreos/etcd/pull/3249)

```go
// 32-bit-atomic-bug.go
package main
import "sync/atomic"

type a struct {
    b chan struct{}
    c int64
}

func main(){
    d := a{}
    atomic.StoreInt64(&d.c, 10 * 1000 * 1000 * 1000)
}
```

```console
$ GOARCH=386 go build 32-bit-atomic-bug.go
$ file 32-bit-atomic-bug
32-bit-atomic-bug: ELF 32-bit LSB executable, Intel 80386, version 1 (SYSV), statically linked, not stripped
$ ./32-bit-atomic-bug
panic: runtime error: invalid memory address or nil pointer dereference
[signal 0xb code=0x1 addr=0x0 pc=0x808cd9b]

goroutine 1 [running]:
panic(0x8098de0, 0x1830a038)
  /usr/local/go/src/runtime/panic.go:481 +0x326
sync/atomic.StoreUint64(0x1830e0f4, 0x540be400, 0x2)
  /usr/local/go/src/sync/atomic/asm_386.s:190 +0xb
main.main()
  /tmp/32-bit-atomic-bug.go:11 +0x4b
```

This means that all structs should keep all `int64` and `uint64` fields at the top of the struct to be safe. If we would move `a.c` to the top of the `a` struct above, the operation would succeed.

The bug affects `32-bit` platforms when a `(u)int64` field is accessed by an `atomic` method.
It would be great to write a tool that checks so all `atomic` accessed fields are aligned at the top of the struct, but it's hard: [coreos/etcd#5027](https://github.com/coreos/etcd/issues/5027).

## Prebuilding the Go standard library (`std`)

A great blog post [that is describing this](https://medium.com/@rakyll/go-1-5-cross-compilation-488092ba44ec#.5jcd0owem)

Before Go 1.5, the whole Go project had to be cross-compiled from source for **all** platforms that _might_ be used, and that was quite a slow process:

```console
# From build/build-image/cross/Dockerfile when we used Go 1.4
$ cd /usr/src/go/src
$ for platform in ${PLATFORMS}; do GOOS=${platform%/*} GOARCH=${platform##*/} ./make.bash --no-clean; done
```

With Go 1.5+, cross-compiling the Go repository isn't required anymore. Go will automatically cross-compile the `std` packages that are being used by the code that is being compiled, _and throw it away after the compilation_.
If you cross-compile multiple times, Go will build parts of `std`, throw it away, compile parts of it again, throw that away and so on.

However, there is an easy way of cross-compiling all `std` packages in advance with Go 1.5+:

```console
# From build/build-image/cross/Dockerfile when we're using Go 1.5+
$ for platform in ${PLATFORMS}; do GOOS=${platform%/*} GOARCH=${platform##*/} go install std; done
```

### Static cross-compilation

Static compilation with Go 1.5+ is dead easy:

```go
// main.go
package main
import "fmt"
func main() {
    fmt.Println("Hello Kubernetes!")
}
```

```console
$ go build main.go
$ file main
main: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked, not stripped
$ GOOS=linux GOARCH=arm go build main.go
$ file main
main: ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), statically linked, not stripped
```

The only thing you have to do is change the `GOARCH` and `GOOS` variables. Here's a list of valid values for [GOOS/GOARCH](https://golang.org/doc/install/source#environment)

#### Static compilation with `net`

Consider this:

```go
// main-with-net.go
package main
import "net"
import "fmt"
func main() {             
	fmt.Println(net.ParseIP("10.0.0.10").String())
}
```

```console
$ go build main-with-net.go
$ file main-with-net
main-with-net: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), dynamically linked, 
    interpreter /lib64/ld-linux-x86-64.so.2, not stripped
$ GOOS=linux GOARCH=arm go build main-with-net.go
$ file main-with-net
main-with-net: ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), statically linked, not stripped
```

Wait, what? Just because we included `net` from the `std` package, the binary defaults to being dynamically linked when the target platform equals to the host platform?
Let's take a look at `go env` to get a clue why this happens:

```console
$ go env
GOARCH="amd64"
GOHOSTARCH="amd64"
GOHOSTOS="linux"
GOOS="linux"
GOPATH="/go"
GOROOT="/usr/local/go"
GO15VENDOREXPERIMENT="1"
CC="gcc"
CXX="g++"
CGO_ENABLED="1"
```

See the `CGO_ENABLED=1` at the end? That's where compilation for the host and cross-compilation differs. By default, Go will link statically if no `cgo` code is involved. `net` is one of the packages that prefers `cgo`, but doesn't depend on it.

When cross-compiling on the other hand, `CGO_ENABLED` is set to `0` by default.

To always be safe, run this when compiling statically:

```console
$ CGO_ENABLED=0 go build -a -installsuffix cgo main-with-net.go
$ file main-with-net
main-with-net: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked, not stripped
```

See [golang/go#9344](https://github.com/golang/go/issues/9344) for more details.

### Dynamic cross-compilation

In order to dynamically compile a go binary with `cgo`, we need `gcc` installed at build time.

The only Kubernetes binary that is using C code is the `kubelet`, or in fact `cAdvisor` on which `kubelet` depends. `hyperkube` is also dynamically linked as long as `kubelet` is. We should aim to make `kubelet` statically linked.

The normal `x86_64-linux-gnu` can't cross-compile binaries, so we have to install gcc cross-compilers for every platform. We do this in the [`kube-cross`](../../build/build-image/cross/Dockerfile) image,
and depend on the [`emdebian.org` repository](https://wiki.debian.org/CrossToolchains). Depending on `emdebian` isn't ideal, so we should consider using the latest `gcc` cross-compiler packages from the `ubuntu` main repositories in the future.

Here's an example when cross-compiling plain C code:

```c
// main.c
#include <stdio.h>
main()
{
  printf("Hello Kubernetes!\n");
}
```

```console
$ arm-linux-gnueabi-gcc -o main-c main.c
$ file main-c
main-c: ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), dynamically linked, 
    interpreter /lib/ld-linux.so.3, for GNU/Linux 2.6.32, not stripped
```

And here's an example when cross-compiling `go` and `c`:

```go
// main-cgo.go
package main
/*
char* sayhello(void) { return "Hello Kubernetes!"; }
*/
import "C"
import "fmt"
func main() {
	fmt.Println(C.GoString(C.sayhello()))
}
```

```console
$ CGO_ENABLED=1 CC=arm-linux-gnueabi-gcc GOOS=linux GOARCH=arm go build main-cgo.go
$ file main-cgo
./main-cgo: ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), dynamically linked, 
    interpreter /lib/ld-linux.so.3, for GNU/Linux 2.6.32, not stripped
```

The bad thing with dynamic compilation is that it adds an unnecessary dependency on `glibc` _at runtime_.

### Static compilation with CGO code

Lastly, it's even possible to cross-compile `cgo` code _statically_:

```console
$ CGO_ENABLED=1 CC=arm-linux-gnueabi-gcc GOARCH=arm go build -ldflags '-extldflags "-static"' main-cgo.go
$ file main-cgo
./main-cgo: ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), statically linked,
    for GNU/Linux 2.6.32, not stripped
```

This is especially useful if we want to include the binary in a container.
If the binary is statically compiled, we may use `busybox` or even `scratch` as the base image.
This should be the preferred way of compiling binaries that strictly require C code to be a part of it.

#### GOARM

32-bit ARM comes in two main flavours: ARMv5 and ARMv7. Go has the `GOARM` environment variable that controls which version of ARM Go should target. Here's a table of all ARM versions and how they play together:

ARM Version | GOARCH | GOARM | GCC package | No. of bits
----------- | ------ | ----- | ----------- | -----------
ARMv5       | arm    | 5     | armel       | 32-bit
ARMv6       | arm    | 6     | -           | 32-bit
ARMv7       | arm    | 7     | armhf       | 32-bit
ARMv8       | arm64  | -     | aarch64     | 64-bit

The compability between the versions is pretty straightforward, ARMv5 binaries may run on ARMv7 hosts, but not vice versa.

## Cross-building docker images for linux

After binaries have been cross-compiled, they should be distributed in some manner.

The default and maybe the most intuitive way of doing this is by packaging it in a docker image.

### Trivial Dockerfile

All `Dockerfile` commands except for `RUN` works for any architecture without any modification.
The base image has to be switched to an arch-specific one, but except from that, a cross-built image is only a `docker build` away.

```Dockerfile
FROM armel/busybox
ENV kubernetes=true
COPY kube-apiserver /usr/local/bin/
CMD ["/usr/local/bin/kube-apiserver"]
```

```console
$ file kube-apiserver
kube-apiserver: ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), statically linked, not stripped
$ docker build -t gcr.io/google_containers/kube-apiserver-arm:v1.x.y .
Step 1 : FROM armel/busybox
 ---> 9bb1e6d4f824
Step 2 : ENV kubernetes true
 ---> Running in 8a1bfcb220ac
 ---> e4ef9f34236e
Removing intermediate container 8a1bfcb220ac
Step 3 : COPY kube-apiserver /usr/local/bin/
 ---> 3f0c4633e5ac
Removing intermediate container b75a054ab53c
Step 4 : CMD /usr/local/bin/kube-apiserver
 ---> Running in 4e6fe931a0a5
 ---> 28f50e58c909
Removing intermediate container 4e6fe931a0a5
Successfully built 28f50e58c909
```

### Complex Dockerfile

However, in the most cases, `RUN` statements are needed when building the image.

The `RUN` statement invokes `/bin/sh` inside the container, but in this example, `/bin/sh` is an ARM binary, which can't execute on an `amd64` processor.

#### QEMU to the rescue

Here's a way to run ARM Docker images on an amd64 host by using `qemu`:

```console
# Register other architectures` magic numbers in the binfmt_misc kernel module, so it`s possible to run foreign binaries
$ docker run --rm --privileged multiarch/qemu-user-static:register --reset
# Download qemu 2.5.0
$ curl -sSL https://github.com/multiarch/qemu-user-static/releases/download/v2.5.0/x86_64_qemu-arm-static.tar.xz \
    | tar -xJ
# Run a foreign docker image, and inject the amd64 qemu binary for translating all syscalls
$ docker run -it -v $(pwd)/qemu-arm-static:/usr/bin/qemu-arm-static armel/busybox /bin/sh

# Now we`re inside an ARM container although we`re running on an amd64 host
$ uname -a
Linux 0a7da80f1665 4.2.0-25-generic #30-Ubuntu SMP Mon Jan 18 12:31:50 UTC 2016 armv7l GNU/Linux
```

Here a linux module called `binfmt_misc` registered the "magic numbers" in the kernel, so the kernel may detect which architecture a binary is, and prepend the call with `/usr/bin/qemu-(arm|aarch64|ppc64le)-static`. For example, `/usr/bin/qemu-arm-static` is a statically linked `amd64` binary that translates all ARM syscalls to `amd64` syscalls.

The multiarch guys have done a great job here, you may find the source for this and other images at [GitHub](https://github.com/multiarch)


## Implementation

## History

32-bit ARM (`linux/arm`) was the first platform Kubernetes was ported to, and luxas' project [`Kubernetes on ARM`](https://github.com/luxas/kubernetes-on-arm) (released on GitHub the 31st of September 2015)
served as a way of running Kubernetes on ARM devices easily.
The 30th of November 2015, a tracking issue about making Kubernetes run on ARM was opened: [#17981](https://github.com/kubernetes/kubernetes/issues/17981). It later shifted focus to how to make Kubernetes a more platform-independent system.

The 27th of April 2016, Kubernetes `v1.3.0-alpha.3` was released, and it became the first release that was able to run the [docker getting started guide](http://kubernetes.io/docs/getting-started-guides/docker/) on `linux/amd64`, `linux/arm`, `linux/arm64` and `linux/ppc64le` without any modification.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/multi-platform.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
