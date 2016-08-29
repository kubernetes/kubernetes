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

We need a way to link "foo" to an owner. Owner references in ConfigMaps
should enable their management. ConfigMaps without owner references will
continue to work as expected today.

```sh
$ kubectl set owner configmap/foo deployment/bar
configmap "foo" updated
```

The ConfigMap needs to be mounted as a volume in the Deployment it references as an owner.

```sh
$ kubectl set volume deployment/bar -t configmap --configmapname foo
deployment "bar" updated
```

A new controller needs to watch for Deployments and ConfigMaps. Whenever it
spots a Deployment with a mounted ConfigMap that has an owner reference
pointing back to the Deployment, it will create a copy of the ConfigMap, 
remove the original owner reference, and mount the copy in the owner. It 
should generally track ConfigMaps with owner references to Deployments, 
create new copies on updates, and mount them back in the owner. The only 
valid owner of a ConfigMap for the controller will be a Deployment for 
now. In the future, the controller can be extended to facilitate config 
management for DaemonSets, PetSets, and any other resource that may need it.

Once the copied ConfigMap is mounted in the Deployment, the deployment
controller needs to create the new ReplicaSet and set the owner reference
for the copy to point to that ReplicaSet. In that way we ensure that copies
of the original ConfigMap will be pruned by the garbage collector once their
owner ReplicaSets are deleted. Certain copies of a ConfigMap will end up 
being shared by more than one ReplicaSets.

Additionally, the deployment controller should not create a new ReplicaSet
for a Deployment "bar" that uses a ConfigMap "foo" with an owner reference
that points back to "bar". Instead, it should wait until the new controller
updates the Deployment pod template with the first copy of "foo".

The original ConfigMap can be programmatically distinguished from its copies
by looking at the owner reference. It should be the only one pointing to a
Deployment. The new controller can also label all copies including the 
original ConfigMap in order to track them as a group.


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

We need a way to specify owner references on ConfigMaps since these refs are going to enable CM management. A generic `set owner`, `set owner-reference`, or `set owner-ref` will be added in `kubectl`. We may also want to add a flag in `create configmap` to set the reference on creation.

```sh
$ kubectl create configmap foo --from-file=config.yml
configmap "foo" created
$ kubectl set owner configmap/foo deployment/bar
configmap "foo" updated
# also maybe
$ kubectl create configmap foo --from-file=config.yml --owner deployment/bar
configmap "foo" created
```

We also need a way to set volumes in pod templates without having to edit files by hand. A `kubectl set volume` command will be added to accommodate that need.

```sh
$ kubectl set volume deployment/bar -t configmap --configmapname foo
deployment "bar" updated
```

## Issues

We need to make sure end-users understand that copies of a ConfigMap are not
supposed to be managed in any way. A controller loop will manage those
similar to how ReplicaSets for a Deployment are managed by the deployment
controller. We may want to disallow mutations on them (can be part of this
proposal or as a follow-up).

## Future

It should be possible to extend the new controller to support Secrets, PVCs,and any other type we may want to trigger on change. Also, it can be reused by other higher-level resources that may need the same triggering mechanism  such as PetSets and DaemonSets. We also probably need to disallow mutations on the copies of the original ConfigMap in order to prevent pods from accidentally picking up wrong configuration on restart.

## References

- https://github.com/kubernetes/kubernetes/issues/22368


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/configmap-rollout.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
