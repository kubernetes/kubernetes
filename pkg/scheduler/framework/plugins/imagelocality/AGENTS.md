# ImageLocality Plugin (`pkg/scheduler/framework/plugins/imagelocality`)

This guide provides an architectural overview, interface implementations, image scoring algorithms, adaptive spread calculations, caching mechanics, and testing strategies for the `ImageLocality` plugin in `pkg/scheduler/framework/plugins/imagelocality`.

---

## 1. High-Level Purpose & Scope

The `ImageLocality` plugin is a scoring plugin that prioritizes worker nodes that already have the container images required by a pod cached locally. Placing pods onto nodes with existing image caches minimizes container startup latency and reduces cluster network bandwidth consumption.

### Core Responsibilities:
1. **Multi-Source Image Inspection**: Evaluates all container images declared in `pod.Spec.Containers`, `pod.Spec.InitContainers`, and image volume sources (`pod.Spec.Volumes[*].Image.Reference`).
2. **CRI-Compliant Image Normalization**: Canonicalizes image strings (e.g., ensuring implicit `:latest` tag consistency).
3. **Image State Cache Lookup**: Queries node-level image cache summaries (`nodeInfo.GetImageStates()`) provided by the scheduler's node cache.
4. **Adaptive Spread Scaling**: Dynamically scales raw image size scores by the image's cluster spread ratio (`NumNodes / TotalNodes`) to mitigate the "node heating problem".
5. **Linear Clamping & Normalization**: Maps aggregate image scores between `minThreshold` (23 MiB) and `maxThreshold` (1000 MiB per container) into the standard scheduler score range (`[0, 100]`).
6. **Opportunistic Batch Signatures**: Implements `SignPlugin` to fingerprint pods by their requested container images for batch scheduling.

---

## 2. Package Architecture & File Map

```
pkg/scheduler/framework/plugins/imagelocality/
├── image_locality.go       # Plugin definition, Score, SignPod, calculation and scaling algorithms
├── image_locality_test.go  # Comprehensive table-driven unit tests for multi-node scoring and edge cases
└── AGENTS.md               # This agent documentation
```

---

## 3. Data Structures & Key Constants

### 3.1. Constants

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `Name` | `names.ImageLocality` (`"ImageLocality"`) | Registered plugin name in scheduler profiles. |
| `minThreshold` | `23 * 1024 * 1024` (23 MiB) | Lower bound below which image size contributions are clamped to 0. |
| `maxContainerThreshold` | `1000 * 1024 * 1024` (1000 MiB) | Maximum score contribution threshold per container/image source. |
| `fwk.ImageNamesSignerName` | `"ImageNamesSigner"` | Signature fragment key used for pod grouping. |

---

## 4. Extension Point Implementations

`ImageLocality` implements `fwk.ScorePlugin` and `fwk.SignPlugin`.

```
                    ┌───────────────────────────────┐
                    │          Incoming Pod         │
                    └───────────────┬───────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           │                                                 │
           ▼                                                 ▼
┌─────────────────────────────────┐               ┌─────────────────────────────────┐
│ SignPod (SignPlugin)            │               │ Score (ScorePlugin)             │
│ - Collect all container images  │               │ - Retrieve total cluster nodes  │
│ - Normalize image names         │               │ - Sum scaled image scores       │
│ - Return fwk.ImageNamesSigner   │               │ - Calculate bounded priority    │
└─────────────────────────────────┘               └────────────────┬────────────────┘
                                                                   │
                                                                   ▼
                                                  ┌─────────────────────────────────┐
                                                  │ Return Score in [0, 100]        │
                                                  └─────────────────────────────────┘
```

### 4.1. `Score` (`ScorePlugin`)
- **Signature**: `Score(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status)`
- **Workflow**:
  1. Retrieves total node count from `pl.handle.SnapshotSharedLister().NodeInfos().List()`.
  2. Invokes `sumImageScores(nodeInfo, pod, totalNumNodes)` to aggregate weighted image sizes and total image count.
  3. Invokes `calculatePriority(imageScores, imageCount)` to compute the final node score in range `[0, fwk.MaxScore]`.

### 4.2. `SignPod` (`SignPlugin`)
- **Signature**: `SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status)`
- **Behavior**: Extracts and normalizes image names from `pod.Spec.Containers` and `pod.Spec.InitContainers`, returning sorted image names as a `fwk.SignFragment`.

---

## 5. Scoring & Caching Algorithms

```
                          ┌───────────────────────────────┐
                          │   Node Cached Image State     │
                          │ (nodeInfo.GetImageStates())   │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  For each image in Pod Spec:          │
                      │  1. Normalize name                    │
                      │  2. Lookup ImageStateSummary          │
                      │     (Size, NumNodes)                  │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  scaledImageScore:                    │
                      │  spread = NumNodes / TotalNodes       │
                      │  score = Size * spread                │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  sumScores = sum(scaledImageScore)    │
                      │  maxThreshold = 1000MB * imageCount   │
                      │  clamp sumScores to [23MB, maxThresh] │
                      │  score = 100 * (sum - min) / (max - min)
                      └───────────────────────────────────────┘
```

### 5.1. Image State Summary & Node Caching
The scheduler maintains an in-memory cache of image metadata on each node (`fwk.ImageStateSummary`):
- `Size`: The uncompressed/decompressed byte size of the image.
- `NumNodes`: The total number of nodes in the cluster that currently have this image cached.

### 5.2. Adaptive Spread Scaling & "Node Heating" Mitigation
If score was purely based on raw size, a massive container image cached on only a single node would heavily bias all new pods towards that single node, causing resource bottlenecks ("node heating problem").

To prevent this, `scaledImageScore` dampens the score of images that exist on few nodes:
$$\text{spread} = \frac{\text{imageState.NumNodes}}{\text{totalNumNodes}}$$
$$\text{scaledScore} = \lfloor \text{imageState.Size} \times \text{spread} \rfloor$$

As an image spreads across more nodes, its cache hit score increases proportionately on all nodes holding it.

### 5.3. Priority Clamping & Normalization
The sum of scaled scores is clamped between:
- $\text{minThreshold} = 23\text{ MiB}$
- $\text{maxThreshold} = 1000\text{ MiB} \times \text{imageCount}$

$$\text{FinalScore} = \frac{100 \times (\text{sumScores} - \text{minThreshold})}{\text{maxThreshold} - \text{minThreshold}}$$

---

## 6. Test Fixtures & Unit Testing

`image_locality_test.go` verifies scoring behavior:
- **Image Size Range Tests**: Verifies clamping below 23 MiB (yields 0 score) and above maximum container thresholds (yields 100 score).
- **Spread Factor Tests**: Validates that an image present on 2 out of 10 nodes scores lower than the same image present on 8 out of 10 nodes.
- **Image Volumes**: Tests image references in `pod.Spec.Volumes[*].Image.Reference`.
- **Image Normalization**: Tests image name tag completion (e.g. `gcr.io/foo` normalized to `gcr.io/foo:latest`).
