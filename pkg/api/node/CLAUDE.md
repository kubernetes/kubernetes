# Package: node

## Purpose
Provides utilities for handling deprecated node labels and generating warnings for Node-related API objects that use deprecated configurations.

## Key Variables
- `deprecatedNodeLabels` - Map of deprecated label keys to deprecation messages (e.g., `beta.kubernetes.io/arch` -> use `kubernetes.io/arch`)

## Key Functions
- `GetNodeLabelDeprecatedMessage(key string) (string, bool)` - Returns deprecation message for a label key if deprecated
- `GetWarningsForRuntimeClass(rc *node.RuntimeClass) []string` - Warns about deprecated labels in RuntimeClass node selectors
- `GetWarningsForNodeSelector(nodeSelector *metav1.LabelSelector, fieldPath *field.Path) []string` - Checks matchExpressions and matchLabels for deprecated labels
- `GetWarningsForNodeSelectorTerm(nodeSelectorTerm api.NodeSelectorTerm, checkLabelValue bool, fieldPath *field.Path) []string` - Validates node selector terms for deprecated labels and optionally invalid label values

## Deprecated Labels Tracked
- `beta.kubernetes.io/arch` and `beta.kubernetes.io/os` (since v1.14)
- `failure-domain.beta.kubernetes.io/region` and `zone` (since v1.17)
- `beta.kubernetes.io/instance-type` (since v1.17)
- `node-role.kubernetes.io/master` (use control-plane instead)
