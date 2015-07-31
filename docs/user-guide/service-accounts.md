<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/service-accounts.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Service Accounts

A service account provides an identity for processes that run in a Pod.

*This is a user introduction to Service Accounts.  See also the
[Cluster Admin Guide to Service Accounts](../admin/service-accounts-admin.md).*

*Note: This document describes how service accounts behave in a cluster set up
as recommended by the Kubernetes project.  Your cluster administrator may have
customized the behavior in your cluster, in which case this documentation may
not apply.*

When you (a human) access the cluster (e.g. using kubectl), you are
authenticated by the apiserver as a particular User Account (currently this is
usually "admin", unless your cluster administrator has customized your
cluster).  Processes in containers inside pods can also contact the apiserver.
When they do, they are authenticated as a particular Service Account (e.g.
"default").

## Using the Default Service Account to access the API server.

When you create a pod, you do not need to specify a service account.  It is
automatically assigned the `default` service account of the same namespace.  If
you get the raw json or yaml for a pod you have created (e.g. `kubectl get
pods/podname -o yaml`), you can see the `spec.serviceAccount` field has been
[automatically set](working-with-resources.md#resources-are-automatically-modified).

You can access the API using a proxy or with a client library, as described in
[Accessing the Cluster](accessing-the-cluster.md#accessing-the-api-from-a-pod).

## Using Multiple Service Accounts

Every namespace has a default service account resource called "default".
You can list this and any other serviceAccount resources in the namespace with this command:

```console
$ kubectl get serviceAccounts
NAME      SECRETS
default   1
```

You can create additional serviceAccounts like this:

```console
$ cat > /tmp/serviceaccount.yaml <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: build-robot
EOF
$ kubectl create -f /tmp/serviceaccount.yaml
serviceaccounts/build-robot
```

If you get a complete dump of the service account object, like this:

```console
$ kubectl get serviceaccounts/build-robot -o yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  creationTimestamp: 2015-06-16T00:12:59Z
  name: build-robot
  namespace: default
  resourceVersion: "272500"
  selfLink: /api/v1/namespaces/default/serviceaccounts/build-robot
  uid: 721ab723-13bc-11e5-aec2-42010af0021e
secrets:
- name: build-robot-token-bvbk5
```

then you will see that a token has automatically been created and is referenced by the service account.

In the future, you will be able to configure different access policies for each service account.

To use a non-default service account, simply set the `spec.serviceAccount`
field of a pod to the name of the service account you wish to use.

The service account has to exist at the time the pod is created, or it will be rejected.

You cannot update the service account of an already created pod.

You can clean up the service account from this example like this:

```console
$ kubectl delete serviceaccount/build-robot
```

<!-- TODO: describe how to create a pod with no Service Account. -->

## Adding Secrets to a service account.

TODO: Test and explain how to use additional non-K8s secrets with an existing service account.

TODO explain:
  - The token goes to: "/var/run/secrets/kubernetes.io/serviceaccount/$WHATFILENAME"


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/service-accounts.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
