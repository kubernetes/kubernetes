When a test suite like test/e2e/e2e.test from Kubernetes includes this
package, the `-storage.testdriver` parameter can be used one or more
times to enabling testing of a certain pre-installed storage driver.

The parameter takes as argument the name of a .yaml or .json file. The
filename can be absolute or relative to `--repo-root`. The content of
the file is used to populate a struct that defines how to test the
driver. For a full definition of the content see:
- `struct driverDefinition` in [external.go](./external.go)
- `struct TestDriver` and the `Cap` capability constants in [testdriver.go](../framework/testdriver.go)

Here is a minimal example for the CSI hostpath driver:

    StorageClass:
      FromName: true
    SnapshotClass:
      FromName: true
    DriverInfo:
      Name: hostpath.csi.k8s.io
      Capabilities:
        persistence: true

The `prow.sh` script of the different CSI hostpath driver releases
generates the actual definition that is used during CI testing, for
example in
[v1.2.0](https://github.com/kubernetes-csi/csi-driver-host-path/blob/v1.2.0/release-tools/prow.sh#L748-L763).

Currently there is no checking for unknown fields, i.e. only file
entries that match with struct entries are used and other entries are
silently ignored, so beware of typos.

For each driver, the storage tests from `test/e2e/storage/testsuites`
are added for that driver with `External Storage [Driver: <Name>]` as
prefix.

To run just those tests for the example above, put that content into
`/tmp/hostpath-testdriver.yaml`, ensure `e2e.test` is in your PATH or current directory (downloaded from a test tarball like https://dl.k8s.io/release/v1.14.0/kubernetes-test-linux-amd64.tar.gz or built via `make WHAT=test/e2e/e2e.test`), and invoke:

    ginkgo -p -focus='External.Storage.*hostpath.csi.k8s.io' \
           -skip='\[Feature:|\[Disruptive\]' \
           e2e.test \
           -- \
           -storage.testdriver=/tmp/hostpath-testdriver.yaml

This disables tests which depend on optional features. Those tests
must be run by selecting them explicitly in an environment that
supports them, for example snapshotting:

    ginkgo -p -focus='External.Storage.*hostpath.csi.k8s.io.*\[Feature:VolumeSnapshotDataSource\]' \
           -skip='\[Disruptive\]' \
           e2e.test \
           -- \
           -storage.testdriver=/tmp/hostpath-testdriver.yaml
