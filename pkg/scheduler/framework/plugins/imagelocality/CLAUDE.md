# Package: imagelocality

## Purpose
Implements a scoring plugin that favors nodes that already have the container images required by a pod. Reduces pod startup time by avoiding image pulls.

## Key Types

### ImageLocality
The plugin struct implementing `fwk.ScorePlugin` and `fwk.SignPlugin`:
- **handle**: Framework handle for accessing node information

## Key Constants

- **minThreshold**: 23 MB - minimum image size for scoring consideration
- **maxContainerThreshold**: 1000 MB - maximum per-container image size for scoring

## Key Functions

- **New(ctx, obj, handle)**: Creates a new ImageLocality plugin
- **Score(ctx, state, pod, nodeInfo)**: Returns a score based on cached images
- **SignPod(ctx, pod)**: Returns normalized image names for pod signing

### Scoring Algorithm

1. **sumImageScores(nodeInfo, pod, totalNumNodes)**:
   - Sums image scores for all containers (init + regular + image volumes)
   - Each image's score is its size scaled by spread factor

2. **scaledImageScore(imageState, totalNumNodes)**:
   - Scales image size by (numNodesWithImage / totalNodes)
   - Mitigates "node heating" where pods cluster on nodes with images

3. **calculatePriority(sumScores, numContainers)**:
   - Clamps sumScores to [minThreshold, maxThreshold * numContainers]
   - Normalizes to [0, MaxNodeScore]

### Image Name Handling
- **normalizedImageName(name)**: Appends ":latest" if no tag specified
- Handles CRI-compliant image naming for cache lookups

## Design Pattern
- Scoring only (no filtering) - won't prevent scheduling, just influences node selection
- Anti-hotspot: spread factor reduces preference for widely-cached images
- Supports image volumes (OCI images as volumes) in addition to containers
