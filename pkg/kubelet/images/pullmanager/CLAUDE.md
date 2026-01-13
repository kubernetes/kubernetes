# Package: pullmanager

Manages image pull tracking with credential verification for the KubeletEnsureSecretPulledImages feature.

## Key Types/Structs

- **PullManager**: Implementation of ImagePullManager that tracks images pulled by the kubelet, recording credentials used for each successful pull to enable multi-tenant image access control.
- **ImagePullManager**: Interface for managing image pull state and credential verification.
- **PullRecordsAccessor**: Interface for unified access to ImagePullIntents/ImagePulledRecords storage.
- **GetPodCredentials**: Function type for lazy credential lookup.

## Key Functions

- `NewImagePullManager()`: Creates a PullManager with records accessor and policy enforcer.
- `RecordPullIntent()`: Records intent to pull an image before the pull occurs.
- `RecordImagePulled()`: Records successful image pull with credentials used.
- `RecordImagePullFailed()`: Decrements reference counter when pull fails.
- `MustAttemptImagePull()`: Evaluates whether a pod needs to re-pull an image based on cached credentials and policy.
- `PruneUnknownRecords()`: Garbage collects records for images no longer present.

## Storage Types

- **ImagePullIntent**: Tracks in-flight image pulls (survives kubelet restart).
- **ImagePulledRecord**: Records successfully pulled images with credential mappings per image name.

## Design Notes

- Uses striped locks for concurrent access to intent and pulled records
- Reference counting for concurrent pulls of the same image
- Credentials tracked per image name (without tag/digest) mapped to imageRef
- Supports node-wide access, Kubernetes secrets, and service account credentials
- Merges credentials from multiple sources into cached records
- Initializes on startup by reconciling in-flight intents with CRI image list
