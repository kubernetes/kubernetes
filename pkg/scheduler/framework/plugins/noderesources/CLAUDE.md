# Package: noderesources

## Purpose
Implements plugins for checking and scoring nodes based on resource capacity. Includes NodeResourcesFit for filtering and multiple scoring strategies.

## Key Types

### Fit
The main plugin struct implementing:
- PreFilter, Filter, PreScore, Score, EnqueueExtensions, SignPlugin
- Supports multiple scoring strategies via configuration

### Scoring Strategies
- **LeastAllocated**: Prefers nodes with more available resources
- **MostAllocated**: Prefers nodes with less available resources (bin packing)
- **RequestedToCapacityRatio**: Custom scoring curve based on utilization

### resourceAllocationScorer
Internal scorer implementation with:
- **Name**: Strategy name
- **scorer**: Scoring function
- **resources**: List of resources to consider with weights

## Extension Points

### PreFilter
- Computes pod resource requests (CPU, memory, extended resources)
- Handles in-place vertical scaling if enabled
- Caches computed requests in CycleState

### Filter
- Checks if node has sufficient allocatable resources
- Considers both requested and limit-based resources
- Handles pod-level resources when enabled

### PreScore / Score
- Applies configured scoring strategy
- Normalizes scores to [0, MaxNodeScore]
- Supports resource-specific weights

## Key Functions

- **NewFit(ctx, args, handle, features)**: Creates NodeResourcesFit plugin
- **NewBalancedAllocation(ctx, args, handle, features)**: Creates balanced allocation scorer
- **computePodResourceRequest(pod, features)**: Calculates total pod resource needs

## Configuration (NodeResourcesFitArgs)
- **ScoringStrategy**: Type and parameters for scoring
- **IgnoredResources**: Resources to exclude from checking
- **IgnoredResourceGroups**: Resource groups to exclude

## Design Pattern
- Separates filtering (hard constraint) from scoring (soft preference)
- Configurable scoring enables different packing strategies
- Supports extended resources and DRA resources
