# Package: fuzzer

## Purpose
Provides fuzz testing functions for the apps API group types, used to generate random but valid instances of apps objects for testing serialization roundtrips and API machinery.

## Key Variables
- `Funcs`: A function that returns fuzzer functions for apps API types

## Fuzzer Functions Provided
- `ControllerRevision`: Generates random revisions with valid RawExtension data
- `StatefulSet`: Fuzzes with proper defaults for PodManagementPolicy (OrderedReady), UpdateStrategy (RollingUpdate), PVCRetentionPolicy (Retain), RevisionHistoryLimit (10)
- `Deployment`: Fuzzes with selector matching template labels
- `DeploymentSpec`: Generates random RevisionHistoryLimit and ProgressDeadlineSeconds
- `DeploymentStrategy`: Randomly chooses Recreate or RollingUpdate, generates MaxUnavailable/MaxSurge for rolling updates
- `DaemonSet`: Fuzzes with labels matching template
- `DaemonSetSpec`: Generates random RevisionHistoryLimit
- `DaemonSetUpdateStrategy`: Randomly chooses RollingUpdate or OnDelete
- `ReplicaSet`: Fuzzes with selector matching template labels

## Design Notes
- Ensures fuzzed objects match defaulter behavior for consistency
- Uses randfill library for random value generation
- Critical for API machinery testing and version conversion verification
