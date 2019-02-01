# WARNING: Do not add tests in this directory

There are two types of end-to-end tests in Kubernetes:
 * [Cluster end-to-end tests](https://git.k8s.io/community/contributors/devel/e2e-tests.md)
 * [Node end-to-end
   tests](https://github.com/kubernetes/community/blob/master/contributors/devel/e2e-node-tests.md)

Tests located in `${KUBE_ROOT}/test/e2e/common` are shared by both Cluster
and Node E2E test jobs. Tests in `${KUBE_ROOT}/test/e2e_node` are exclusively
owned by Node E2E. *If you want to add a test, most likely than not, you want
to add the test to one of the two directories mentioned above.* If you are
unsure, please check with the OWNER of the directory.

Tests in this directory are exclusively Cluster E2E tests. Currently, most of
the tests in this directory are misplaced and legacy; and will be cleaned up in
the future.
