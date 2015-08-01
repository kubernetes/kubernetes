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
[here](http://releases.k8s.io/release-1.0/docs/admin/service-accounts-admin.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Cluster Admin Guide to Service Accounts

*This is a Cluster Administrator guide to service accounts.  It assumes knowledge of
the [User Guide to Service Accounts](../user-guide/service-accounts.md).*

*Support for authorization and user accounts is planned but incomplete.  Sometimes
incomplete features are referred to in order to better describe service accounts.*

## User accounts vs service accounts

Kubernetes distinguished between the concept of a user account and a service accounts
for a number of reasons:
  - User accounts are for humans.  Service accounts are for processes, which
    run in pods.
  - User accounts are intended to be global. Names must be unique across all
    namespaces of a cluster, future user resource will not be namespaced).
    Service accounts are namespaced.
  - Typically, a cluster's User accounts might be synced from a corporate
    database, where new user account creation requires special privileges and
    is tied to complex business  processes.  Service account creation is intended
    to be more lightweight, allowing cluster users to create service accounts for
    specific tasks (i.e. principle of least privilege).
  - Auditing considerations for humans and service accounts may differ.
  - A config bundle for a complex system may include definition of various service
    accounts for components of that system.  Because service accounts can be created
    ad-hoc and have namespaced names, such config is portable.

## Service account automation

Three separate components cooperate to implement the automation around service accounts:
  - A Service account admission controller
  - A Token controller
  - A Service account controller

### Service Account Admission Controller

The modification of pods is implemented via a plugin
called an [Admission Controller](admission-controllers.md). It is part of the apiserver.
It acts synchronously to modify pods as they are created or updated. When this plugin is active
(and it is by default on most distributions), then it does the following when a pod is created or modified:
  1. If the pod does not have a `ServiceAccount` set, it sets the `ServiceAccount` to `default`.
  2. It ensures that the `ServiceAccount` referenced by the pod exists, and otherwise rejects it.
  4. If the pod does not contain any `ImagePullSecrets`, then `ImagePullSecrets` of the
`ServiceAccount` are added to the pod.
  5. It adds a `volume` to the pod which contains a token for API access.
  6. It adds a `volumeSource` to each container of the pod mounted at `/var/run/secrets/kubernetes.io/serviceaccount`.

### Token Controller

TokenController runs as part of controller-manager. It acts asynchronously. It:
- observes serviceAccount creation and creates a corresponding Secret to allow API access.
- observes serviceAccount deletion and deletes all corresponding ServiceAccountToken Secrets
- observes secret addition, and ensures the referenced ServiceAccount exists, and adds a token to the secret if needed
- observes secret deleteion and removes a reference from the corresponding ServiceAccount if needed

#### To create additional API tokens

A controller loop ensures a secret with an API token exists for each service
account. To create additional API tokens for a service account, create a secret
of type `ServiceAccountToken` with an annotation referencing the service
account, and the controller will update it with a generated token:

```json
secret.json:
{
	"kind": "Secret",
	"metadata": {
		"name": "mysecretname",
		"annotations": {
			"kubernetes.io/service-account.name": "myserviceaccount"
		}
	}
	"type": "kubernetes.io/service-account-token"
}
```

```sh
kubectl create -f ./secret.json
kubectl describe secret mysecretname
```

#### To delete/invalidate a service account token

```sh
kubectl delete secret mysecretname
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/service-accounts-admin.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
