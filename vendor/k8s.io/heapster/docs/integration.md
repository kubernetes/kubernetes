## Heapster integration testing

Heapster can be tested against an existing Kubernetes cluster running on GCE. Tests under [integration](../integration) can be used for integration testing.

	make test-integration

Note that the test expects Kube Config to exist under `~/.kube/.kubeconfig`. To override the default path using the flag `--kube_config`
