# Package: dynamicresources

## Purpose
Implements the Dynamic Resource Allocation (DRA) plugin for scheduling pods that use ResourceClaims. Handles allocation, reservation, and binding of dynamic resources like GPUs, FPGAs, and other devices.

## Key Types

### DynamicResources
The plugin struct implementing multiple extension points:
- PreEnqueue, PreFilter, Filter, PostFilter, Score, Reserve, PreBind, EnqueueExtensions, SignPlugin
- **enabled**: Whether DRA feature is enabled
- **fts**: Feature flags for DRA-related features
- **filterTimeout/bindingTimeout**: Configurable timeouts
- **draManager**: Shared DRA state manager
- **celCache**: Cache for compiled CEL expressions

### stateData
Per-pod scheduling cycle state:
- **claims**: ResourceClaims for the pod
- **allocator**: Structured allocator for claim allocation
- **unavailableClaims**: Claims that failed on some nodes
- **nodeAllocations**: Per-node allocation results

## Key Functions

- **New(ctx, args, handle, features)**: Creates plugin (no-op if DRA disabled)
- **PreEnqueue(ctx, pod)**: Validates pod's ResourceClaims exist
- **PreFilter(ctx, state, pod)**: Gathers claims, checks if already allocated
- **Filter(ctx, state, pod, nodeInfo)**: Checks if node can satisfy claims
- **PostFilter(ctx, state, pod, statusMap)**: Deallocates claims if scheduling fails
- **Reserve(ctx, state, pod, nodeName)**: Marks claims as allocated for the node
- **PreBind(ctx, state, pod, nodeName)**: Updates ResourceClaim status in API

## Design Pattern
- Disabled by default, enabled via DynamicResourceAllocation feature gate
- Uses CEL for evaluating device selector expressions
- Supports both immediate and delayed allocation modes
- Handles extended resources backed by DRA when DRAExtendedResource feature is enabled
- Thread-safe with mutex for parallel Filter execution
