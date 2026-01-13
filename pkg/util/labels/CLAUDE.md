# Package: labels

## Purpose
Provides utilities for working with Kubernetes labels and label selectors.

## Key Functions
- `CloneAndAddLabel()` - Creates a copy of labels map with a new label added
- `CloneAndRemoveLabel()` - Creates a copy of labels map with a label removed
- `AddLabel()` - Adds a label to an existing map (modifies in place, creates if nil)
- `CloneSelectorAndAddLabel()` - Deep clones a LabelSelector and adds a match label
- `AddLabelToSelector()` - Adds a label to selector's MatchLabels (modifies in place)
- `SelectorHasLabel()` - Checks if selector has a specific label in MatchLabels

## Design Patterns
- Clone functions create new maps to avoid mutating originals
- Empty key parameter returns input unchanged (no-op)
- Handles nil inputs gracefully
- Deep cloning includes MatchExpressions for LabelSelectors
