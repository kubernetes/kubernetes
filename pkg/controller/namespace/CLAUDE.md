# Package: namespace

Kubernetes Namespace controller for namespace lifecycle management.

## Key Types

- `NamespaceController`: Performs actions dependent on namespace phase, primarily handling namespace deletion

## Key Constants

- `namespaceDeletionGracePeriod`: 5 seconds - delay before processing deletion to allow HA apiservers to sync

## Key Functions

- `NewNamespaceController()`: Creates the controller with namespace informer and deletion helper
- `Run()`: Starts the reconciliation loop
- `syncNamespaceFromKey()`: Main reconciliation logic

## Purpose

Manages namespace lifecycle, specifically handling the deletion of namespaces and all resources within them. When a namespace is marked for deletion, this controller coordinates the cleanup of all namespaced resources.

## Key Features

- Delays processing of deletion events to handle HA apiserver race conditions
- Delegates actual resource deletion to NamespacedResourcesDeleter
- Uses rate-limited workqueue with fast retry (5ms to 60s)

## Design Notes

- Only queues namespaces with non-nil DeletionTimestamp
- Grace period allows admission plugins on other apiservers to block new object creation
- Uses tuned rate limiter for reliable namespace cleanup
