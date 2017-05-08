This directory is the staging area for packages that have been split to their
own repository. The content here will be periodically published to respective
top-level k8s.io repositories.

Most code in the `staging/` directory is authoritative, i.e. the only copy of
the code. You can directly modify such code. However the packages in
`staging/src/k8s.io/client-go/pkg` are copied from `pkg/`. If you modify the
original code in `pkg/`, you need to run `hack/godep-restore.sh` from the k8s
root directory, followed by `hack/update-staging-client-go.sh`. We are working
towards making all code in `staging/` authoritative.

The `vendor/k8s.io` directory contains symlinks pointing to this staging area,
so to use a package in the staging area, you can import it as
`k8s.io/<package-name>`, as if the package were vendored. Packages will be
vendored from `k8s.io/<package-name>` for real after the test matrix is
converted to vendor k8s components.
