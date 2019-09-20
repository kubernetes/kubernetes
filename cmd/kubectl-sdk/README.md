# Overview

This kubectl target is the binary that is packaged and distributed through gcloud.
The standard kubectl target (in the cmd/kubectl directory) is dynamically linked
with the BoringCrypto library in order to be FIPS compliant. This kubectl binary
is built statically, and it now includes the dispatcher (shim) functionality.
Sean Sullivan (seans), 9/20/2019.

## To Build:

```
make kubectl-sdk
```

Binary is at (linux/amd64):

```
_output/local/bin/linux/amd64/kubectl-sdk
_output/local/go/bin/kubectl-sdk
```

```
bazel build //cmd/kubectl-sdk:kubectl
```

Binary is at (linux/amd64):

```
bazel-bin/cmd/kubectl-sdk/linux_amd64_pure_stripped/kubectl
```

