<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# ConfigMap management

## Abstract

A proposal for implementing management of ConfigMaps in the context of a
[Deployment](http://kubernetes.io/docs/user-guide/deployments/).

Users should be able to have their apps automatically pick up changes to
their configuration and also have the ability to rollback to previous
configuration similar to the ability to rollback to previous pod template
revisions.

Currently, in order to apply updated configuration into a running app, users
need to 1) either do it manually: copy the ConfigMap that holds their old
behavior, update it, store it back as a different ConfigMap, and mount the
new ConfigMap into the Deployment, or 2) edit the original ConfigMap and
trigger the deployment controller to start a new ReplicaSet (either by
adding a dummy environment value in the pod template of the Deployment or
by including the update as part of another update) losing the ability to
rollback to previous configuration.

This doc proposes a solution for removing the burden of ConfigMap management
from users by providing a mechanism that will work out of the box without
having to manually rollout new ReplicaSets.

## Implementation

### Proposed solution

Users should only ever need to create and update a single ConfigMap "foo".
```sh
$ kubectl create configmap foo --from-file=config.yml
configmap "foo" created
```

We need a way to link "foo" to the Deployment that is going to be triggered
every time "foo" is updated.
```sh
$ kubectl set triggers deployment/bar configmap/foo
deployment "bar" updated
```

A `{resource-0}.kubernetes.io/triggered-by: {resource-1}/{name-1}` annotation
will be added in the resource we want to trigger automatic updates on. In the
example above, a `deployment.kubernetes.io/triggered-by: configmap/foo`
annotation should be added in Deployment "bar". We may want to extended this
mechanism to support cross-namespace triggering since vendors may want to
trigger on various resource updates not necessarily found in the same namespace
as the resource that expects to be triggered.

As today, the ConfigMap needs to be mounted as a volume in the Deployment.
```sh
$ kubectl set volume deployment/bar -t configmap --configmapname foo
deployment "bar" updated
```

A new controller needs to watch for Deployments and ConfigMaps. Whenever it
spots a Deployment with a `triggered-by` annotation and a mounted ConfigMap,
it will create a copy of the ConfigMap and mount the copy in the Deployment.
It should generally track Deployments with `triggered-by` annotations, 
create new ConfigMap copies on updates of the original ConfigMap, and mount
them back in the Deployment. Only Deployments should work for this controller
for now but in the future the controller can be extended to facilitate config 
management for DaemonSets, PetSets, and any other resource that may need it.

Once the copied ConfigMap is mounted in the Deployment, the deployment
controller needs to create the new ReplicaSet and set the owner reference
for the copy to point to the ReplicaSet. In that way we ensure that copies
of the original ConfigMap will be pruned by the garbage collector once their
owner ReplicaSets are deleted. Certain copies of a ConfigMap will end up 
being shared by more than one ReplicaSets.

Additionally, the deployment controller should not create a new ReplicaSet
for a Deployment "bar" that holds a `triggered-by` annotation to a ConfigMap
"foo". Instead, it should wait until the new controller updates the Deployment
template with the first copy of "foo". This ensures that the original ConfigMap
copy will not land in a ReplicaSet, hence will not be subject to garbage
collection.


### Alternative solutions

* Embed ConfigMap as a template in the Deployment.

While it will work smoothly on ConfigMap changes for Deployments, this
solution will need to be spread across the API for all resources that face
the same problem. Sooner or later, higher-level resources will get bloated
trying to manage stuff like ConfigMaps, Secrets, et al.

* Others?

## Changes

### API

No API changes will be needed for Deployments or ConfigMaps.

### Controller

A new controller will be needed for watching Deployments and Config Maps. Changes will also be needed in the deployment controller. [See above](#proposed-solution).

### CLI

We need a way to specify relationship between a resource that is updated (ConfigMap in our case) and a resource that will be triggered on those updates (Deployment). A generic `set trigger` will be added in `kubectl`. The relationship will be established with an annotation on the triggered resource (Deployment).
```sh
$ kubectl set trigger deployment/bar configmap/foo
deployment "bar" updated
$ kubectl get deployment/bar -o yaml | grep triggered-by
deployment.kubernetes.io/triggered-by: configmap/foo
```

We also need a way to set volumes in pod templates without having to edit files by hand. A `kubectl set volume` command will be added to accommodate that need.
```sh
$ kubectl set volume deployment/bar -t configmap --configmapname foo
deployment "bar" updated
```

## Issues

* We need to make sure end-users understand that copies of a ConfigMap are not
supposed to be managed in any way. A controller loop will manage those
similarly to how ReplicaSets for a Deployment are managed by the deployment
controller. Eventually, the garbage collector should prune them once they go
out of scope ie. no running pod is using them and all of their owners are
deleted.

* We cannot rollback the original ConfigMap when a Deployment is rolled back
because it may be used by other Deployments and rolling back would break
those. How confusing will it be for a user to rollback her Deployment and
have the original ConfigMap ie. the one she uses to trigger rollouts hold
configuration different from what she is currently running?

* We may want to disallow mutations on copies of the original ConfigMap (can
be part of this proposal or as a follow-up).

## Future

It should be possible to extend the new controller to support Secrets, PVCs,
and any other type we may want to trigger on update. Also, it can be reused
by other higher-level resources that may need the same triggering mechanism
such as PetSets and DaemonSets. We also probably need to disallow mutations
on the copies of the original ConfigMap in order to prevent pods from
accidentally picking up wrong configuration on restart.

## References

- https://github.com/kubernetes/kubernetes/issues/22368


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/configmap-rollout.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
