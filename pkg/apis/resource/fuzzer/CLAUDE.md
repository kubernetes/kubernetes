# Package: fuzzer

## Purpose
Provides fuzzer functions for the resource API group to generate valid random test data for fuzz testing DRA types.

## Key Functions

### Funcs
Returns fuzzer functions for resource types:

- **ExactDeviceRequest fuzzer**: Ensures AllocationMode is set (randomly to All or ExactCount)
- **DeviceSubRequest fuzzer**: Same as ExactDeviceRequest for subrequests
- **DeviceAllocationConfiguration fuzzer**: Ensures Source is set (FromClass or FromClaim)
- **DeviceToleration fuzzer**: Ensures Operator is set (Equal or Exists)
- **DeviceTaint fuzzer**: Sets TimeAdded to current time truncated to seconds
- **OpaqueDeviceConfiguration fuzzer**: Sets Parameters to valid JSON matching runtime.Object default
- **AllocatedDeviceStatus fuzzer**: Sets Data to valid JSON matching runtime.Object default
- **ResourceSliceSpec fuzzer**: Normalizes AllNodes (false -> nil) and NodeName (empty -> nil)

## Notes
- Truncates time to seconds for round-trip compatibility
- Uses stable JSON format for RawExtension fields to prevent re-encoding changes
- Critical for testing DRA API stability with random inputs
