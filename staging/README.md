# External Repository Staging Area

This directory is the staging area for packages that have been split to their
own repository. The content here will be periodically published to respective
top-level k8s.io repositories.

Repositories currently staged here:

- [`k8s.io/api`](https://github.com/kubernetes/api)
- [`k8s.io/apiextensions-apiserver`](https://github.com/kubernetes/apiextensions-apiserver)
- [`k8s.io/apimachinery`](https://github.com/kubernetes/apimachinery)
- [`k8s.io/apiserver`](https://github.com/kubernetes/apiserver)
- [`k8s.io/cli-runtime`](https://github.com/kubernetes/cli-runtime)
- [`k8s.io/client-go`](https://github.com/kubernetes/client-go)
- [`k8s.io/cloud-provider`](https://github.com/kubernetes/cloud-provider)
- [`k8s.io/cluster-bootstrap`](https://github.com/kubernetes/cluster-bootstrap)
- [`k8s.io/code-generator`](https://github.com/kubernetes/code-generator)
- [`k8s.io/component-base`](https://github.com/kubernetes/component-base)
- [`k8s.io/component-helpers`](https://github.com/kubernetes/component-helpers)
- [`k8s.io/controller-manager`](https://github.com/kubernetes/controller-manager)
- [`k8s.io/cri-api`](https://github.com/kubernetes/cri-api)
- [`k8s.io/csi-translation-lib`](https://github.com/kubernetes/csi-translation-lib)
- [`k8s.io/dynamic-resource-allocation`](https://github.com/kubernetes/dynamic-resource-allocation)
- [`k8s.io/endpointslice`](https://github.com/kubernetes/endpointslice)
- [`k8s.io/kms`](https://github.com/kubernetes/kms)
- [`k8s.io/kube-aggregator`](https://github.com/kubernetes/kube-aggregator)
- [`k8s.io/kube-controller-manager`](https://github.com/kubernetes/kube-controller-manager)
- [`k8s.io/kube-proxy`](https://github.com/kubernetes/kube-proxy)
- [`k8s.io/kube-scheduler`](https://github.com/kubernetes/kube-scheduler)
- [`k8s.io/kubectl`](https://github.com/kubernetes/kubectl)
- [`k8s.io/kubelet`](https://github.com/kubernetes/kubelet)
- [`k8s.io/legacy-cloud-providers`](https://github.com/kubernetes/legacy-cloud-providers)
- [`k8s.io/metrics`](https://github.com/kubernetes/metrics)
- [`k8s.io/mount-utils`](https://github.com/kubernetes/mount-utils)
- [`k8s.io/pod-security-admission`](https://github.com/kubernetes/pod-security-admission)
- [`k8s.io/sample-apiserver`](https://github.com/kubernetes/sample-apiserver)
- [`k8s.io/sample-cli-plugin`](https://github.com/kubernetes/sample-cli-plugin)
- [`k8s.io/sample-controller`](https://github.com/kubernetes/sample-controller)

The code in the staging/ directory is authoritative, i.e. the only copy of the
code. You can directly modify such code.

## Using staged repositories from Kubernetes code

Kubernetes code uses the repositories in this directory via a Go workspace and
module `replace` statements.  For example, when Kubernetes code imports a
package from the `k8s.io/client-go` repository, that import is resolved to
`staging/src/k8s.io/client-go` relative to the project root:

```go
// pkg/example/some_code.go
package example

import (
  "k8s.io/client-go/dynamic" // resolves to staging/src/k8s.io/client-go/dynamic
)
```

## Creating a new repository in staging

### Adding the staging repository in `kubernetes/kubernetes`:

1. Send an email to the SIG Architecture [mailing
   list](https://groups.google.com/forum/#!forum/kubernetes-sig-architecture)
   and the mailing list of the SIG which would own the repo requesting approval
   for creating the staging repository.

2. Once approval has been granted, create the new staging repository.

3. Update
   [`import-restrictions.yaml`](/staging/publishing/import-restrictions.yaml)
   to add the list of other staging repos that this new repo can import.

4. Add all mandatory template files to the staging repo as mentioned in
   https://github.com/kubernetes/kubernetes-template-project.

5. Make sure that the `.github/PULL_REQUEST_TEMPLATE.md` and `CONTRIBUTING.md`
   files mention that PRs are not directly accepted to the repo.

6. Ensure that `docs.go` file is added. Refer to
   [#kubernetes/kubernetes#91354](https://github.com/kubernetes/kubernetes/blob/release-1.24/staging/src/k8s.io/client-go/doc.go)
   for reference.

7. NOTE: Do not edit go.mod or go.sum in the new repo (staging/src/k8s.io/<newrepo>/) manually. Run the following instead:

```
  ./hack/update-vendor.sh
```

8. Run [`./hack/update-go-workspace.sh`](/hack/update-go-workspace.sh) to add
   the module to the workspace.

### Creating the published repository

1. Create an [issue](https://github.com/kubernetes/org/issues/new?template=repo-create.md)
in the `kubernetes/org` repo to request creation of the respective published
repository in the Kubernetes org. The published repository **must** have an
initial empty commit. It also needs specific access rules and branch settings.
See [#kubernetes/org#58](https://github.com/kubernetes/org/issues/58)
for an example.

2. Setup branch protection and enable access to the `stage-bots` team
by adding the repo in
[`prow/config.yaml`](https://github.com/kubernetes/test-infra/blob/master/config/prow/config.yaml).
See [#kubernetes/test-infra#9292](https://github.com/kubernetes/test-infra/pull/9292)
for an example.

3. Once the repository has been created in the Kubernetes org,
update the publishing-bot to publish the staging repository by updating:

    - [`rules.yaml`](/staging/publishing/rules.yaml):
    Make sure that the list of dependencies reflects the staging repos in the `Godeps.json` file.

    - [`repos.sh`](https://github.com/kubernetes/publishing-bot/blob/master/hack/repos.sh):
    Add the staging repo in the list of repos to be published.

4. Add the staging and published repositories as a subproject for the
SIG that owns the repos in
[`sigs.yaml`](https://github.com/kubernetes/community/blob/master/sigs.yaml).

5. Add the repo to the list of staging repos in this `README.md` file.
