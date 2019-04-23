When a test suite like test/e2e/e2e.test from Kubernetes includes this
package, the -storage.testdriver parameter can be used one or more
times to enabling testing of a certain pre-installed storage driver.

The parameter takes as argument the name of a .yaml or .json file. The
filename can be absolute or relative to --repo-root. The content of
the file is used to populate a struct that defines how to test the
driver. For a full definition of the struct see the external.go file.

Here is an example for the CSI hostpath driver:

    ShortName: mytest
    StorageClass:
      FromName: true
    SnapshotClass:
      FromName: true
    DriverInfo:
      Name: csi-hostpath
      Capabilities:
        persistence: true
        dataSource: true
        multipods: true

Currently there is no checking for unknown fields, i.e. only file
entries that match with struct entries are used and other entries are
silently ignored, so beware of typos.

For each driver, the storage tests from `test/e2e/storage/testsuites`
are added for that driver with `External Storage [Driver: <Name>]` as
prefix.

To run just those tests for the example above, put that content into
`/tmp/hostpath-testdriver.yaml`, ensure `e2e.test` is in your PATH or current directory (downloaded from a test tarball like https://storage.googleapis.com/kubernetes-release/release/v1.14.0/kubernetes-test-linux-amd64.tar.gz or built via `make WHAT=test/e2e/e2e.test`), and invoke:

    ginkgo -p -focus='External.Storage.*csi-hostpath' \
           -skip='\[Feature:|\[Disruptive\]' \
           e2e.test \
           -- \
           -storage.testdriver=/tmp/hostpath-testdriver.yaml

This disables tests which depend on optional features. Those tests
must be run by selecting them explicitly in an environment that
supports them, for example snapshotting:

    ginkgo -p -focus='External.Storage.*csi-hostpath.*\[Feature:VolumeSnapshotDataSource\]' \
           -skip='\[Disruptive\]' \
           e2e.test \
           -- \
           -storage.testdriver=/tmp/hostpath-testdriver.yaml
