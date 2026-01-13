# Package: storageversionmigrator

Implements the StorageVersionMigration controller that migrates stored resources to a new storage version.

## Key Types

- **SVMController**: Processes StorageVersionMigration resources to re-encode stored objects.

## Key Functions

- **NewSVMController**: Creates controller with dynamic client and GC graph builder.
- **Run**: Starts workers processing migration requests.
- **sync**: Main sync logic - validates conditions and triggers migration.
- **runMigration**: Iterates through resources and applies patches to trigger re-encoding.
- **failMigration**: Marks migration as failed with error message.

## Key Constants

- **migrationSuccessStatusReason**: `StorageVersionMigrationSucceeded`
- **migrationRunningStatusReason**: `StorageVersionMigrationInProgress`
- **migrationFailedStatusReason**: `StorageVersionMigrationFailed`

## Key Features

- Uses garbage collector's resource monitors for consistent resource listing.
- Applies server-side apply patches to trigger storage re-encoding.
- Handles conflicts gracefully (resource updated/deleted during migration).
- Supports retry for transient errors (throttling, timeouts).
- Validates resource version ordering to avoid processing stale data.

## Design Patterns

- Depends on GC's GraphBuilder for resource monitoring.
- Uses dynamic client for resource-agnostic operations.
- Skips already migrated resources based on resource version comparison.
- Updates status conditions to track migration progress.
