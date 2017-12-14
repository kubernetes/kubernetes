TODO: This document was manually maintained so might be incomplete. The
automation effort is tracked in https://github.com/kubernetes/test-infra/issues/5843.

# kubernetes-1.8.4

**Bug fixes and Improvements**:

* The kube-apiserver handles empty patch request correctly (e.g. by a no-op kubectl apply).

    * [https://github.com/kubernetes/kubernetes/pull/54780](https://github.com/kubernetes/kubernetes/pull/54780)

# kubernetes-1.8.2

**Bug fixes and Improvements**:

* Fix memory leak in kube-apiserver with CustomResourceDefinitions.

    * [https://github.com/kubernetes/kubernetes/pull/53586](https://github.com/kubernetes/kubernetes/pull/53586)

* Fix error message for validation of API version of CustomResources.

    * [https://github.com/kubernetes/kubernetes/pull/54218](https://github.com/kubernetes/kubernetes/pull/54218)

# kubernetes-1.8.0

**Action Required**:

* The deprecated `ThirdPartyResource` (TPR) API was removed. To avoid losing your TPR data, [migrate to CustomResourceDefinition](https://kubernetes.io/docs/tasks/access-kubernetes-api/migrate-third-party-resource/).

    * [https://github.com/kubernetes/kubernetes/pull/48353](https://github.com/kubernetes/kubernetes/pull/48353)

**New Features**:

* [alpha] The CustomResourceDefinition API can now optionally validate custom objects based on a JSON schema provided in the CRD spec. Enable this alpha feature with the CustomResourceValidation feature gate in kube-apiserver.

    * [https://github.com/kubernetes/kubernetes/pull/47263](https://github.com/kubernetes/kubernetes/pull/47263)

* CustomResourceDefinitions support `metadata.generation` and implement spec/status split.

    * [https://github.com/kubernetes/kubernetes/pull/50764](https://github.com/kubernetes/kubernetes/pull/50764)
