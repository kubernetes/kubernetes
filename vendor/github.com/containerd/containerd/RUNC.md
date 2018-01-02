containerd is built with OCI support and with support for advanced features provided by [runc](https://github.com/opencontainers/runc).

We depend on a specific `runc` version when dealing with advanced features.  You should have a specific runc build for development.  The current supported runc commit is:

RUNC_COMMIT = 74a17296470088de3805e138d3d87c62e613dfc4

For more information on how to clone and build runc see the runc Building [documentation](https://github.com/opencontainers/runc#building).

Note: before building you may need to install additional support, which will vary by platform. For example, you may need to install `libseccomp` and `libapparmor` e.g. `libseccomp-dev` and `libapparmor-dev` for Ubuntu.

## building

From within your `opencontainers/runc` repository run:

### apparmor

```bash
make BUILDTAGS='seccomp apparmor' && sudo make install
```

### selinux

```bash
make BUILDTAGS='seccomp selinux' && sudo make install
```

After an official runc release we will start pinning containerd support to a specific version but various development and testing features may require a newer runc version than the latest release.  If you encounter any runtime errors, please make sure your runc is in sync with the commit/tag provided in this document.
