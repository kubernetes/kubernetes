# Package: testing

## Purpose
Provides test utilities and a fake API server reactor (VolumeReactor) for testing the PersistentVolume controller.

## Key Types

- **VolumeReactor**: Simulates etcd and API server for testing. Stores volumes/claims, tracks changes, supports fake watchers, and can inject errors.
- **ReactorError**: Defines errors to inject for specific verb/resource combinations.

## Key Functions

### Reactor Lifecycle
- **NewVolumeReactor(ctx, client, fakeVolumeWatch, fakeClaimWatch, errors)**: Creates a reactor with fake client reactors.
- **React(ctx, action)**: Main callback handling create/update/get/delete operations with version checking.

### State Management
- **AddVolume/AddVolumes**: Add PVs to reactor state.
- **AddClaim/AddClaims**: Add PVCs to reactor state.
- **DeleteVolume**: Remove a PV by name.
- **MarkVolumeAvailable**: Reset a PV to Available state.
- **AddClaimBoundToVolume**: Add a PVC and bind its corresponding PV.

### Event Simulation
- **DeleteVolumeEvent/DeleteClaimEvent**: Simulate deletion events sent to controller.
- **AddClaimEvent**: Simulate claim addition events.

### Verification
- **CheckVolumes(expectedVolumes)**: Compare expected vs actual volumes.
- **CheckClaims(expectedClaims)**: Compare expected vs actual claims.
- **PopChange(ctx)**: Pop next changed object from queue for verification.
- **SyncAll()**: Enqueue all objects for simulated periodic sync.
- **GetChangeCount()**: Get count of changes since last sync.

## Design Notes

- Simulates resource version conflicts (ErrVersionConflict) like real etcd.
- Supports watch interface for informer-based tests.
- Error injection allows testing error handling paths.
- Thread-safe with RWMutex protection.
