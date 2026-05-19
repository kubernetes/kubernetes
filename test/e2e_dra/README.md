This directory contains a testsuite with automatic upgrade/downgrade tests for
DRA. Conceptually this is like an integration test, in the sense that it
starts/stops cluster components and runs tests against them.

The difference is that it starts Kubernetes components by running the actual
binaries, relying on local-up-cluster.sh for the logic and configuration
steps. Because local-up-cluster.sh needs additional permissions and
preparations on the host, the test cannot run in "make test-integration" and
just skips itself there.

To run it:
- Make sure that hack/local-up-cluster.sh works:
  - sudo must work
  - Set env variables as necessary for your environment.
- Ensure that /var/lib/kubelet/plugins, /var/lib/kubelet/plugins_registry,
  and /var/run/cdi are writable.
- Build binaries with `make`.
- Export `KUBERNETES_SERVER_BIN_DIR=$(pwd)/_output/local/bin/linux/amd64` (or
  whatever is your GOOS/GOARCH and output directory).
- Optional: export `KUBERNETES_SERVER_CACHE_DIR=$(pwd)/_output/local/bin/linx/amd64/cache-dir`
  to reuse downloaded release binaries across test invocations.
- Optional: set ARTIFACTS to store component log files persistently.
  Otherwise a test tmp directory is used.
- Invoke as a Go test (no need for the ginkgo CLI), for example:

        go test -v -count=1 -timeout=1h ./test/e2e_dra -args -ginkgo.v
        dlv test ./test/e2e_dra -- -ginkgo.v
        make test KUBE_TIMEOUT=-timeout=1h WHAT=test/e2e_dra FULL_LOG=true KUBE_TEST_ARGS="-count=1 -args -ginkgo.v"

`make test` instead of `make test-integration` is intentional: `local-up-cluster.sh`
itself wants to start etcd. `-count=1` ensures that test runs each time it is invoked.
`-v` and `-ginkgo.v` make the test output visible while the test runs.

To simplify starting from scratch, `./test/e2e_dra/run.sh` cleans
up, sets permissions, and then invokes whatever command is specified on the
command line:

     ./test/e2e_dra/run.sh go test ./test/e2e_dra

The test is implemented as a Ginkgo suite because that allows reusing the same
helper code as in E2E tests. Long-term the goal is to port that helper code to
ktesting, support ktesting in test/e2e, and turn this test into a normal Go
test.
