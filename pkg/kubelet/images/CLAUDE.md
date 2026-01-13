# Package: images

Manages container image lifecycle including pulling, garbage collection, and credential handling.

## Key Types/Structs

- **imageManager**: Handles image pulling with backoff, credential lookup, and event recording.
- **ImageManager**: Interface for ensuring images exist on the node.
- **realImageGCManager**: Implements ImageGCManager for garbage collecting unused images.
- **ImageGCManager**: Interface for image garbage collection operations.
- **ImageGCPolicy**: Configuration for GC thresholds (high/low percent, min/max age).
- **imageRecord**: Tracks image metadata (first detected, last used, size, pinned status).

## Key Functions

### Image Manager
- `NewImageManager()`: Creates image manager with configurable parallelism and throttling.
- `EnsureImageExists()`: Main entry point - ensures image is present, pulling if needed.
- `pullImage()`: Pulls image with backoff, credential handling, and metrics.

### Image GC Manager
- `NewImageGCManager()`: Creates GC manager with policy validation.
- `Start()`: Begins async image detection and cache updates.
- `GarbageCollect()`: Frees disk space when usage exceeds high threshold.
- `DeleteUnusedImages()`: Removes all unused images (for eviction).
- `freeSpace()`: Deletes images until enough space is freed.

## Error Types

- `ErrImagePullBackOff`: Pull failed, backing off
- `ErrImageNeverPull`: Image absent with Never pull policy
- `ErrImageInspect`: Unable to inspect image
- `ErrInvalidImageName`: Cannot parse image name

## Design Notes

- Supports serial or parallel image pulling with configurable limits
- Implements exponential backoff for failed pulls
- GC orders images by last used time, respects pinned images
- Integrates with credential providers (secrets, service accounts, external plugins)
- Tracks KubeletEnsureSecretPulledImages feature for credential-aware image access
