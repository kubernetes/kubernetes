# Package config

Package config implements pod configuration readers that merge multiple pod sources (apiserver, file, http) into a unified stream of pod updates for the kubelet.

## Key Types

- `PodConfig`: Configuration mux that merges multiple pod sources into ordered updates
- `podStorage`: In-memory pod state storage with source tracking
- `SourcesReady`: Interface for tracking when all pod sources have been seen
- `mux`: Internal multiplexer for combining source channels

## PodConfig Methods

- `NewPodConfig(recorder, observer)`: Creates a new pod configuration merger
- `Channel(ctx, source)`: Creates or returns a channel for a named source
- `Updates()`: Returns channel of denormalized pod updates
- `SeenAllSources(seenSources)`: Checks if all configured sources have sent updates

## Pod Sources

- `apiserver.go`: Watches pods from Kubernetes API server via informer
- `file.go`: Watches static pod manifests from filesystem (file_linux.go, file_unsupported.go)
- `http.go`: Polls static pod manifests from HTTP URL

## SourcesReady Interface

- `AddSource(source)`: Registers a new source
- `AllReady()`: Returns true when all sources have sent at least one SET

## Update Types (from kubetypes)

- SET: Full replacement of pods from source
- ADD/UPDATE/DELETE/REMOVE: Incremental changes
- RECONCILE: Reconciliation updates

## Design Notes

- Pods merged per-source, then combined into unified view
- Redundant updates filtered to minimize downstream work
- Events recorded for pod additions from each source
- Startup SLI observer tracks pod observation times
