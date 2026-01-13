# Package: userns

## Purpose
The `userns` package manages Linux user namespace allocations for pods. It allocates and tracks UID/GID mappings for pods that run with user namespaces enabled (`hostUsers: false`).

## Key Types/Structs

- **UsernsManager**: Manages user namespace allocations. Uses a bitmap allocator to track used ranges and maps pod UIDs to their allocated host ID ranges.
- **userNamespace**: Configuration struct containing UID and GID mappings for a pod's user namespace.
- **idMapping**: Single ID mapping with HostId, ContainerId, and Length fields.

## Key Functions

- **MakeUserNsManager**: Creates a new UsernsManager. Validates configuration, reads existing allocations from disk, and initializes the allocation bitmap.
- **GetOrCreateUserNamespaceMappings**: Returns user namespace configuration for a pod. Creates new allocation if needed, or reads existing from disk.
- **Release**: Releases the user namespace allocation for a pod.
- **CleanupOrphanedPodUsernsAllocations**: Reconciles allocations with running pods and frees orphaned allocations.
- **EnabledUserNamespacesSupport**: Returns whether the UserNamespacesSupport feature gate is enabled.

## Internal Functions

- **allocateOne**: Finds and allocates a free user namespace range.
- **record**: Records an existing user namespace allocation.
- **writeMappingsToFile/readMappingsFromFile**: Persist/restore user namespace config to/from pod directory.
- **parseUserNsFileAndRecord**: Parses stored config and records the allocation.

## Constants

- **userNsUnitLength**: 65536 - standard user namespace size
- **mapReInitializeThreshold**: 1000 - recreate map after this many removals to free memory
- **mappingsFile**: "userns" - filename for persisted mappings

## Design Notes

- Requires UserNamespacesSupport feature gate to be enabled.
- User namespace length must be a multiple of 65536.
- Kubelet mapping ID must be >= userNsLength (can't map root to 0).
- UID and GID mappings must be identical.
- Allocations are persisted to disk for crash recovery.
- Windows is not supported (separate stub implementation).
- Validates runtime handler supports user namespaces before allocation.
