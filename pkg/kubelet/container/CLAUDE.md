# Package container

Package container defines the core container runtime abstraction and related types for the kubelet to interact with container runtimes via CRI.

## Key Interfaces

- `Runtime`: Main container runtime interface (SyncPod, KillPod, GetPods, etc.)
- `ImageService`: Image operations (PullImage, GetImageRef, ListImages, RemoveImage)
- `StreamingRuntime`: Streaming endpoints (exec, attach, port-forward)
- `Attacher`: Container attach operations
- `CommandRunner`: Execute commands in containers

## Key Types

- `Pod`: Group of containers with ID, name, namespace, containers, sandboxes
- `Container`: Runtime container with ID, name, image, state, resources
- `ContainerID`: Runtime-specific container identifier (type://id format)
- `ImageSpec`: Image specification with ID, runtime handler, annotations
- `PodStatus`: Pod status including container statuses and IPs
- `RuntimeStatus`: Overall runtime status and conditions

## Runtime Interface Key Methods

- `SyncPod`: Syncs running pod to desired state
- `KillPod`: Kills all containers in a pod
- `GetPods`: Lists pods (optionally including dead containers)
- `GarbageCollect`: Removes dead containers per GC policy
- `GetPodStatus`: Retrieves pod and container status
- `GetContainerLogs`: Streams container logs
- `CheckpointContainer`: Creates container checkpoint

## Supporting Types

- `RunContainerOptions`: Container runtime configuration (envs, mounts, devices)
- `GCPolicy`: Garbage collection settings (min age, max containers)
- `SyncResult/PodSyncResult`: Results from sync operations
- `Version`: Runtime version comparison interface

## Sub-packages

- `testing/`: Fake runtime, cache, and helper implementations

## Design Notes

- Thread-safety required from Runtime implementations
- Container states: Created, Running, Exited, Unknown
- Supports in-place pod vertical scaling
- CDI devices for hardware access
