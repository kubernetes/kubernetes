## Supported platforms

### Current support

The following table lists etcd support status for common architectures and operating systems,

| Architecture | Operating System | Status       | Maintainers      |
| ------------ | ---------------- | ------------ | ---------------- |
| amd64        | Darwin           | Experimental | etcd maintainers | 
| amd64        | Linux            | Stable       | etcd maintainers |
| amd64        | Windows          | Experimental |                  |
| arm64        | Linux            | Experimental | @glevand         |
| arm          | Linux            | Unstable     |                  |
| 386          | Linux            | Unstable     |                  |

* etcd-maintainers are listed in https://github.com/coreos/etcd/blob/master/MAINTAINERS.

Experimental platforms appear to work in practice and have some platform specific code in etcd, but do not fully conform to the stable support policy. Unstable platforms have been lightly tested, but less than experimental. Unlisted architecture and operating system pairs are currently unsupported; caveat emptor.

### Supporting a new platform

For etcd to officially support a new platform as stable, a few requirements are necessary to ensure acceptable quality:

1. An "official" maintainer for the platform with clear motivation; someone must be responsible for taking care of the platform.
2. Set up CI for build; etcd must compile.
3. Set up CI for running unit tests; etcd must pass simple tests.
4. Set up CI (TravisCI, SemaphoreCI or Jenkins) for running integration tests; etcd must pass intensive tests.
5. (Optional) Set up a functional testing cluster; an etcd cluster should survive stress testing.

### 32-bit and other unsupported systems

etcd has known issues on 32-bit systems due to a bug in the Go runtime. See the [Go issue][go-issue] and [atomic package][go-atomic] for more information.

To avoid inadvertently running a possibly unstable etcd server, `etcd` on unstable or unsupported architectures will print a warning message and immediately exit if the environment variable `ETCD_UNSUPPORTED_ARCH` is not set to the target architecture.

Currently only the amd64 architecture is officially supported by `etcd`.

[go-issue]: https://github.com/golang/go/issues/599
[go-atomic]: https://golang.org/pkg/sync/atomic/#pkg-note-BUG
