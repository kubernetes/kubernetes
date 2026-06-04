This directory contains a testsuite with automatic upgrade/downgrade tests for
DRA. Conceptually this is like an integration test, in the sense that it
starts/stops cluster components and runs tests against them. It has its own
directory because it needs to be started differently than other integration
tests or unit tests, which makes it more like an E2E suite.

The difference is that it starts Kubernetes components by running the actual
binaries, relying on local-up-cluster.sh for the logic and configuration
steps. local-up-cluster.sh needs additional permissions and
preparations on the host.

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

        go test -v -count=1 -timeout=1h ./test/e2e_dra
        dlv test ./test/e2e_dra -- -test.v
        make test KUBE_TIMEOUT=-timeout=1h WHAT=test/e2e_dra FULL_LOG=true KUBE_TEST_ARGS="-count=1"

`make test` instead of `make test-integration` is intentional: `local-up-cluster.sh`
itself wants to start etcd. `-count=1` ensures that test runs each time it is invoked.
`-v`/`-test.v`/`FULL_LOG=true` make the test output visible while the test runs.

To simplify starting from scratch, `./test/e2e_dra/run.sh` cleans
up, sets permissions, and then invokes whatever command is specified on the
command line:

     ./test/e2e_dra/run.sh go test ./test/e2e_dra
