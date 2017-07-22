This directory is the staging area for packages that have been split to their
own repository. The content here will be periodically published to respective
top-level k8s.io repositories.

The code in the `staging/` directory is authoritative, i.e. the only copy of
the code. You can directly modify such code.

The `vendor/k8s.io` directory contains symlinks pointing to this staging area,
so to use a package in the staging area, you can import it as
`k8s.io/<package-name>`, as if the package were vendored. Packages will be
vendored from `k8s.io/<package-name>` for real after the test matrix is
converted to vendor k8s components.
