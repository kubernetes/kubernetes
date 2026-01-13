# Package: manager

## Purpose
The `manager` package defines interfaces for managing cached objects (like Secrets and ConfigMaps) that are referenced by pods.

## Key Interfaces

- **Manager**: Interface for registering/unregistering pods and retrieving objects by namespace and name.
  - `GetObject(namespace, name string)`: Retrieves an object from cache.
  - `RegisterPod(pod *v1.Pod)`: Registers all objects referenced by the pod.
  - `UnregisterPod(pod *v1.Pod)`: Unregisters objects no longer needed by any pod.

- **Store**: Interface for a reference-counted object cache used by cache-based managers.
  - `AddReference(namespace, name, referencedFrom)`: Adds a reference from a pod to an object.
  - `DeleteReference(namespace, name, referencedFrom)`: Removes a reference (object deleted when refcount reaches zero).
  - `Get(namespace, name)`: Retrieves an object from the store.

## Implementations

The package has two main implementations (in separate files):
- **cacheBasedManager**: Uses a Store with reference counting.
- **watchBasedManager**: Uses watches to keep objects up-to-date.

## Design Notes

- RegisterPod/UnregisterPod must be idempotent.
- RegisterPod/UnregisterPod should be efficient and not block on network operations.
- Store implementations use reference counting - objects are only removed when all references are deleted.
- Used by kubelet to manage Secrets and ConfigMaps referenced by pods.
