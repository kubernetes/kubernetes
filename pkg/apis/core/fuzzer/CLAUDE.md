# Package: fuzzer

## Purpose
Provides fuzzer functions for the core API types, used in round-trip testing to ensure API objects can be serialized and deserialized correctly without data loss.

## Key Functions

- **Funcs**: Returns a slice of fuzzer functions for core types that generate valid random test data.

## Fuzzer Functions Provided

Customized fuzzers for types that need special handling:
- **resource.Quantity**: Generates valid decimal quantities.
- **ObjectReference**: Randomizes API version, kind, namespace, name.
- **PodExecOptions, PodAttachOptions**: Ensures Stdout/Stderr are true.
- **PodSpec**: Sets defaults like TerminationGracePeriodSeconds, SecurityContext, SchedulerName.
- **Container, EphemeralContainer**: Sets TerminationMessagePath and Policy.
- **VolumeSource**: Ensures exactly one field is set.
- **Secret**: Sets type to Opaque.
- **PersistentVolume/Claim**: Randomizes phase, reclaim policy, volume mode.
- **Service**: Ensures at least one port, valid session affinity.
- **Various volume sources**: RBD, ISCSI, AzureDisk, ScaleIO with valid defaults.

## Design Notes

- Uses randfill library for generating random data.
- Essential for API round-trip testing in Kubernetes.
- Ensures generated objects match defaulting behavior to pass round-trip tests.
