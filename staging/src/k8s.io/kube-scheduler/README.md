# kube-scheduler
## Coming Soon!

Implements https://github.com/luxas/community/blob/master/keps/sig-cluster-lifecycle/0014-20180707-componentconfig-api-types-to-staging.md#kube-scheduler-changes

It provides
* Provide a versioned API for configuring kube-scheduler.

## Compatibility

HEAD of this repo will match HEAD of k8s.io/apiserver, k8s.io/apimachinery, and k8s.io/client-go.

## Where does it come from?

`kube-scheduler` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/kube-scheduler.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.
