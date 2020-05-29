# containers-golang

[![CircleCI](https://circleci.com/gh/seccomp/containers-golang.svg?style=shield)](https://circleci.com/gh/seccomp/containers-golang)

`containers-golang` is a set of Go libraries used by container runtimes to generate and load seccomp mappings into the kernel.

seccomp (short for secure computing mode) is a BPF based syscall filter language and present a more conventional function-call based filtering interface that should be familiar to, and easily adopted by, application developers.

## Building
   make - Generates seccomp.json file, which contains the whitelisted syscalls that can be used by container runtime engines like [CRI-O][cri-o], [Buildah][buildah], [Podman][podman] and [Docker][docker], and container runtimes like OCI [Runc][runc] to controll the syscalls available to containers.

### Supported build tags

   `seccomp`
   
## Contributing

When developing this library, please use `make` (or `make … BUILDTAGS=…`) to take advantage of the tests and validation.

## Contact

- IRC: #[containers](irc://irc.freenode.net:6667/#containers) on freenode.net

[cri-o]:   https://github.com/kubernetes-incubator/cri-o/pulls
[buildah]: https://github.com/projectatomic/buildah
[podman]:  https://github.com/projectatomic/podman
[docker]:  https://github.com/docker/docker
[runc]:    https://github.com/opencontainers/runc

