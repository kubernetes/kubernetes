# Package: storage/fuzzer

## Purpose
Provides fuzzer functions for generating random storage API objects during testing, used by the Kubernetes API fuzzing framework.

## Key Functions
- `Funcs`: Returns fuzzer functions for the storage API group that generate randomized StorageClass and CSIDriver objects

## Key Behaviors
- StorageClass fuzzer: Randomizes ReclaimPolicy (Delete/Retain) and VolumeBindingMode (Immediate/WaitForFirstConsumer)
- CSIDriver fuzzer: Randomizes VolumeLifecycleModes with 7 different cases including nil, empty, invalid, Persistent, Ephemeral, and combinations
- CSIDriver defaults are applied: AttachRequired=true, PodInfoOnMount=false, StorageCapacity=false, FSGroupPolicy=ReadWriteOnceWithFSType, etc.

## Design Notes
- Uses the randfill library for random value generation
- Fuzzer functions are registered with the codec factory for use in round-trip testing
- Ensures default values are properly applied to match API server defaulting behavior
