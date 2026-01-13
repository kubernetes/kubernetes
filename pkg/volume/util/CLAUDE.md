# Package: util

## Purpose
Provides common utility functions used across volume plugins for path handling, mount operations, volume metrics, SELinux support, and various helper operations.

## Key Types/Structs
- `AtomicWriter` - Writes files atomically using temp directory and rename
- Various helper structs for volume operations

## Key Functions
- `IsReady/SetReady` - Check/set volume readiness via marker file
- `GetSecretForPV` - Retrieves secrets for volume authentication
- `LoadPodFromFile` - Loads pod spec from disk
- `MountOptionFromSpec` - Extracts mount options from volume spec
- `UnmountViaEmptyDir` - Delegates teardown to emptydir wrapper
- `JoinMountOptions` - Combines mount option lists
- `GetPath` - Safely gets path from mounter
- `CalculateTimeoutForVolume` - Computes operation timeout based on volume size

## Design Patterns
- Centralized utilities to avoid code duplication across plugins
- Platform-specific implementations (Linux, Windows, unsupported)
- AtomicWriter ensures atomic file updates for ConfigMap/Secret
- SELinux mount context management utilities
- Storage class helper functions
