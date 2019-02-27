# Notes to run sig-windows tests

1. Prereqs:

```bash
KUBECONFIG=path/to/kubeconfig
curl https://raw.githubusercontent.com/e2e-win/e2e-win-prow-deployment/master/repo-list -o repo_list
export KUBE_TEST_REPO_LIST=$(pwd)/repo_list
```

1. Run only sig-windows tests:

    ```bash
    ./e2e.test --provider=local --ginkgo.noColor --ginkgo.focus="\[sig-windows\]" --node-os-distro="windows"
    ```


# e2e_node/density_test diff

This test is borrowed from the density test in e2e_node/density_test. All but the first test were omitted as well as some logging.
