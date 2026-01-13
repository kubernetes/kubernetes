# Package: nodedeclaredfeatures

## Purpose
Implements a plugin that checks if nodes have all the features required by a pod. Uses the Node Declared Features (NDF) framework to match pod requirements against node capabilities.

## Key Types

### NodeDeclaredFeatures
The plugin struct implementing:
- PreFilter, Filter, EnqueueExtensions, SignPlugin
- **ndfFramework**: Framework for inferring and matching features
- **version**: Kubernetes version for feature inference
- **enabled**: Whether the NodeDeclaredFeatures feature gate is enabled

### preFilterState
Cached state containing:
- **reqs**: Set of features required by the pod

## Extension Points

### PreFilter
- Infers feature requirements from pod spec using NDF framework
- Skips if no requirements inferred (returns Skip status)
- Caches requirements in CycleState

### Filter
- Checks if node's declared features satisfy pod requirements
- Uses NDF MatchNodeFeatureSet for matching
- Returns UnschedulableAndUnresolvable with unsatisfied requirements on failure

## Key Functions

- **New(ctx, args, handle, features)**: Creates plugin (no-op if feature disabled)
- **SignPod(ctx, pod)**: Returns inferred feature requirements for signing
- **EventsToRegister()**: Returns Node/Add, Node/UpdateNodeDeclaredFeature, and Pod/Update events
- **isSchedulableAfterNodeChange**: Queueing hint for node feature changes
- **isSchedulableAfterPodUpdate**: Queueing hint for pod spec changes

## Design Pattern
- Disabled by default via NodeDeclaredFeatures feature gate
- Uses component-helpers/nodedeclaredfeatures library for inference
- Feature requirements are inferred from pod spec (not explicitly declared)
- Supports dynamic re-evaluation when node features change
