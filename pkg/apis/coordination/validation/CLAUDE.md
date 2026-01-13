# Package: validation

## Purpose
Provides validation logic for Lease and LeaseCandidate resources in the coordination.k8s.io API group.

## Key Functions

- **ValidateLease(lease)**: Validates a Lease object including metadata and spec.
- **ValidateLeaseUpdate(lease, oldLease)**: Validates updates to a Lease object.
- **ValidateLeaseSpec(spec, fldPath)**: Validates LeaseSpec fields (duration > 0, transitions >= 0, strategy validity).
- **ValidateLeaseCandidate(lease)**: Validates a LeaseCandidate object.
- **ValidateLeaseCandidateUpdate(lease, oldLease)**: Validates updates, ensuring LeaseName is immutable.
- **ValidateLeaseCandidateSpec(spec, fldPath)**: Validates LeaseCandidateSpec including semver format for versions.
- **ValidateCoordinatedLeaseStrategy(strategy, fldPath)**: Validates strategy is either a known Kubernetes strategy or a qualified name.
- **ValidLeaseCandidateName(name, prefix)**: Validates LeaseCandidate names using ConfigMapKey rules.

## Validation Rules

- LeaseDurationSeconds must be > 0
- LeaseTransitions must be >= 0
- PreferredHolder requires Strategy to be set
- BinaryVersion and EmulationVersion must be valid semver
- BinaryVersion must be >= EmulationVersion
- EmulationVersion required when strategy is OldestEmulationVersion
- LeaseName is immutable on update
