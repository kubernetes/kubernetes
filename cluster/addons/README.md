# Cluster add-ons

## Overview

Cluster add-ons are resources like Services and Deployments (with pods) that are
shipped with the Kubernetes binaries and are considered an inherent part of the
Kubernetes clusters.

There are currently two classes of add-ons:
- Add-ons that will be reconciled.
- Add-ons that will be created if they don't exist.

More details could be found in [addon-manager/README.md](addon-manager/README.md).

## Cooperating Horizontal / Vertical Auto-Scaling with "reconcile class addons"

"Reconcile" class addons will be periodically reconciled to the original state given
by the initial config. In order to make Horizontal / Vertical Auto-scaling functional,
the related fields in config should be left unset. More specifically, leave `replicas`
in `ReplicationController` / `Deployment` / `ReplicaSet` unset for Horizontal Scaling,
leave `resources` for container unset for Vertical Scaling. The periodic reconcile
won't clobbered these fields, hence they could be managed by Horizontal / Vertical
Auto-scaler.

## Add-on naming

The suggested naming for most of the resources is `<basename>` (with no version number).
Though resources like `Pod`, `ReplicationController` and `DaemonSet` are exceptional.
It would be hard to update `Pod` because many fields in `Pod` are immutable. For
`ReplicationController` and `DaemonSet`, in-place update may not trigger the underlying
pods to be re-created. You probably need to change their names during update to trigger
a complete deletion and creation.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/README.md?pixel)]()
