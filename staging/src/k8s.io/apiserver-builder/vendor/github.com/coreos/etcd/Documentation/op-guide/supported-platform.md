## Supported platform

### 32-bit and other unsupported systems

etcd has known issues on 32-bit systems due to a bug in the Go runtime. See #[358][358] for more information.

To avoid inadvertently running a possibly unstable etcd server, `etcd` on unsupported architectures will print
a warning message and immediately exit if the environment variable `ETCD_UNSUPPORTED_ARCH` is not set to
the target architecture.

Currently only the amd64 architecture is officially supported by `etcd`.

[358]: https://github.com/coreos/etcd/issues/358

