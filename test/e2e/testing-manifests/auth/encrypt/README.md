# Encryption at rest testing manifests

This directory contains manifests for testing encryption at rest with a [mock KMS provider](../../../../../staging/src/k8s.io/kms/internal/plugins/_mock). The mock KMS provider is a fake KMS provider that does not communicate with any external KMS. It is used for testing purposes only.

## run-e2e.sh

The `run-e2e.sh` script does the following:

1. Installs required prerequisites: [`kind`](https://sigs.k8s.io/kind) and [`kubetest2`](https://github.com/kubernetes-sigs/kubetest2).
2. Builds the `e2e.test`, `ginkgo` and `kubectl` binaries.
3. Creates local registry if not already present. This registry is used to push the kms mock plugin image.
4. Build and push the kms mock plugin image to the local registry.
5. Connect local registry to kind network so that kind cluster created using `kubetest2` in prow CI job can pull the kms mock plugin image.
6. Create kind cluster using `kubetest2` and run e2e tests.
7. Collect logs and metrics from kind cluster.
8. Delete kind cluster.

The script extracts runtime configurations through environment variables. The following environment variables are supported:

| Variable              | Description                                                                     | Default |
| --------------------- | ------------------------------------------------------------------------------- | ------- |
| `SKIP_DELETE_CLUSTER` | If set to `true`, the kind cluster will not be deleted after the tests are run. | `false` |
| `SKIP_RUN_TESTS`      | If set to `true`, the tests will not be run.                                    | `false` |
| `SKIP_COLLECT_LOGS`   | If set to `true`, the logs and metrics will not be collected.                   | `false` |

### Running the script locally

Run the script locally with the following command:

```bash
test/e2e/testing-manifests/auth/encrypt/run-e2e.sh
```

### Create a local cluster with mock KMS provider

The `run-e2e.sh` script can be used to create a local cluster with mock KMS provider. The following command creates a local cluster with mock KMS provider:

```bash
SKIP_RUN_TESTS=true SKIP_DELETE_CLUSTER=true SKIP_COLLECT_LOGS=true test/e2e/testing-manifests/auth/encrypt/run-e2e.sh
```

Delete the cluster after use:

```bash
kind delete cluster --name=kms
```
