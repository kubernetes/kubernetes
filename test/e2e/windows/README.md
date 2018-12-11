# Notes to run sig-windows tests

1. Prereqs:
    * KUBECONFIG=path/to/kubeconfig
    * curl https://raw.githubusercontent.com/e2e-win/e2e-win-prow-deployment/master/repo-list.txt -o repo_list.yaml
    * export KUBE_TEST_REPO_LIST=$(pwd)/repo_list.yaml

1. Run only sig-windows tests:

    ```./e2e.test --provider=local --ginkgo.noColor --ginkgo.focus=.*sig-windows*```
