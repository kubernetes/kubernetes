> ⚠️ **This is an automatically published [staged repository](https://git.k8s.io/kubernetes/staging#external-repository-staging-area) for Kubernetes**.   
> Contributions, including issues and pull requests, should be made to the main Kubernetes repository: [https://github.com/kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).  
> This repository is read-only for importing, and not used for direct contributions.  
> See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

# apiextensions

The `apiextensions.k8s.io` API types (`CustomResourceDefinition`, `ConversionReview`)
and their generated clientset, listers, informers and apply configurations.

It only depends on `k8s.io/apimachinery` and `k8s.io/client-go`, so programs that
work with CustomResourceDefinitions don't pull in the server implementation in
[`k8s.io/apiextensions-apiserver`](https://github.com/kubernetes/apiextensions-apiserver).

## Migrating from k8s.io/apiextensions-apiserver

The packages under `k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1`,
`k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1` and
`k8s.io/apiextensions-apiserver/pkg/client` are deprecated aliases of the
packages with the same path in this module. `go fix -inline ./...` rewrites
references to their types, constants and functions. Package-level variables such
as `SchemeGroupVersion` and the clientset's `Scheme` and `Codecs` have to be
updated by hand.

`AddToScheme` and `SchemeBuilder` in this module register the types and their
defaulting functions. The ones in `k8s.io/apiextensions-apiserver` also register
the conversions to the internal version, which only API servers need, so API
servers should keep using those.

## Compatibility

HEAD of this repo will match HEAD of k8s.io/apimachinery and k8s.io/client-go.

## Where does it come from?

`apiextensions` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apiextensions.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.

## Community, discussion, contribution, and support

apiextensions is maintained by [SIG API Machinery](https://github.com/kubernetes/community/tree/master/sig-api-machinery).

You can reach the maintainers of this project at:

- Slack: [#sig-api-machinery](https://kubernetes.slack.com/messages/sig-api-machinery)
- Mailing List: [kubernetes-sig-api-machinery](https://groups.google.com/a/kubernetes.io/g/sig-api-machinery)

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
