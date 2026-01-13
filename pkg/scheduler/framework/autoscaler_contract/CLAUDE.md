# Package: contract

## Purpose
Contains contract tests that verify the scheduler framework's public interfaces remain stable. These tests ensure that downstream projects (like cluster-autoscaler) that depend on the framework interfaces won't break due to API changes.

## Key Contract Tests

### Framework Contract
Verifies that the `framework.Framework` interface includes essential methods:
- **RunPreFilterPlugins**: Runs PreFilter extension point
- **RunFilterPlugins**: Runs Filter extension point
- **RunReservePluginsReserve**: Runs Reserve extension point

### Lister Contracts
Verifies that lister interfaces match expected signatures:
- **NodeInfoLister**: List, Get, HavePodsWithAffinityList, HavePodsWithRequiredAntiAffinityList
- **StorageInfoLister**: IsPVCUsedByPods
- **SharedLister**: NodeInfos, StorageInfos
- **ResourceSliceLister**: ListWithDeviceTaintRules
- **DeviceClassLister**: List, Get
- **ResourceClaimTracker**: List, Get, ListAllAllocatedDevices, and claim management methods
- **SharedDRAManager**: ResourceClaims, ResourceSlices, DeviceClasses, DeviceClassResolver

### CycleState Contract
Verifies that `framework.NewCycleState()` returns the expected type.

## Important Notes
- This package contains only test files (*_test.go) - no production code
- Tests use interface satisfaction checks (var _ Interface = &Implementation{})
- Breaking these tests indicates a breaking API change that affects downstream consumers
- Changes to this package's interfaces require coordination with cluster-autoscaler and other consumers
