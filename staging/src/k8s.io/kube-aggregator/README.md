# kube-aggregator

Implements the [Aggregated API Servers](https://github.com/kubernetes/design-proposals-archive/blob/main/api-machinery/aggregated-api-servers.md) design proposal.

It provides:

* an API for registering API servers.
* Summaries of discovery information from all the aggregated APIs
* HTTP proxying of requests from clients on to specific API backends


## Purpose

We want to divide the single monolithic API server into multiple aggregated
servers. Anyone should be able to write their own aggregated API server to expose APIs they want.
Cluster admins should be able to expose new APIs at runtime by bringing up new
aggregated servers.


## Compatibility

HEAD of this repo will match HEAD of k8s.io/apiserver, k8s.io/apimachinery, and k8s.io/client-go.

## Where does it come from?

`kube-aggregator` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/kube-aggregator.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.

