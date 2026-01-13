# Package: history

ControllerRevision management for StatefulSet and DaemonSet history.

## Key Constants

- `ControllerRevisionHashLabel`: "controller.kubernetes.io/hash" - label for revision hash

## Key Types

- `Interface`: Interface for managing ControllerRevision history (list, create, update, delete, adopt, release)
- `realHistory`: Production implementation using Kubernetes client
- `fakeHistory`: Testing implementation

## Key Functions

- `NewControllerRevision()`: Creates a ControllerRevision with proper labels and owner references
- `HashControllerRevision()`: Computes FNV hash of revision data for deduplication
- `SortControllerRevisions()`: Sorts revisions by revision number
- `EqualRevision()`: Compares two revisions for semantic equality
- `FindEqualRevisions()`: Finds revisions matching a given revision's data

## Purpose

Provides utilities for managing ControllerRevision objects, which store historical versions of controller specs (like StatefulSet or DaemonSet templates). Used to implement rollback functionality.

## Design Notes

- Revision names are formatted as `{controller-name}-{hash}`
- Hash collisions are handled via collision count parameter
- Revisions are sorted by revision number, then creation time, then name
- Implements adopt/release pattern for controller reference management
