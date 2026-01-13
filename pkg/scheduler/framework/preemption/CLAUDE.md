# Package: preemption

## Purpose
Implements pod preemption logic for the Kubernetes scheduler, allowing higher-priority pods to evict lower-priority pods when resources are constrained.

## Key Types
- `Evaluator` - Main interface for evaluating preemption candidates
- `Candidate` - Represents a potential preemption victim with the node and pods to evict
- `candidateList` - Thread-safe list of preemption candidates

## Key Functions
- `NewEvaluator()` - Creates a new preemption evaluator
- `Preempt()` - Main entry point for finding preemption candidates
- `SelectCandidate()` - Chooses the best preemption candidate from a list
- `GetOffsetAndNumCandidates()` - Determines how many candidates to evaluate based on cluster size

## Design Patterns
- Uses parallel evaluation of preemption candidates across nodes
- Implements PDB (PodDisruptionBudget) aware preemption
- Supports dry-run mode for calculating potential preemption without executing
- Uses interface-based design for extensibility with different preemption strategies
