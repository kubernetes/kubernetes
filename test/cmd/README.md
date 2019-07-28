# Kubernetes Command-Line Integration Test Suite

This document describes how you can use the Kubernetes command-line integration test-suite.

## Running Tests

### All Tests

To run this entire suite, execute `make test-cmd` from the top level.  This will import each file containing tests functions

### Specific Tests

To run a subset of tests (e.g. `run_deployment_test` and `run_impersonation_test`), execute `make test-cmd WHAT="deployment impersonation"`.  Running specific
tests will not try and validate any required resources are available on the server.

## Adding Tests

Test functions need to have the format `run_*_test` so they can executed individually.  Once a test has been added, insert a section in `legacy-script.sh` like

```bash
######################
# Replica Sets       #
######################

if kube::test::if_supports_resource "${replicasets}" ; then
    record_command run_rs_tests
fi
```

Be sure to validate any supported resouces required for the test by using the `kube::test::if_supports_resource` function. 


### New File

If the test resides in a new file, source the file in the top of the `legacy-script.sh` file by adding a new line in
```bash
source "${KUBE_ROOT}/test/cmd/apply.sh"
source "${KUBE_ROOT}/test/cmd/apps.sh"
source "${KUBE_ROOT}/test/cmd/authorization.sh"
source "${KUBE_ROOT}/test/cmd/batch.sh"
...
```

Please keep the order of the source list alphabetical.
