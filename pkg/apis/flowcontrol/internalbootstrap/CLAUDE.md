# Package: internalbootstrap

## Purpose
Provides internal (unversioned) representations of mandatory FlowSchema and PriorityLevelConfiguration objects that must exist in every cluster.

## Key Variables

- **MandatoryFlowSchemas**: Map of mandatory FlowSchema objects by name (internal types).
- **MandatoryPriorityLevelConfigurations**: Map of mandatory PriorityLevelConfiguration objects by name (internal types).

## Key Functions

- **NewAPFScheme()**: Creates a new runtime.Scheme with flowcontrol types registered.
- **internalizeFSes(exts)**: Converts v1 FlowSchemas to internal type.
- **internalizePLs(exts)**: Converts v1 PriorityLevelConfigurations to internal type.

## Design Notes

- Converts from bootstrap.MandatoryFlowSchemas and bootstrap.MandatoryPriorityLevelConfigurations (v1 types).
- Used by validation to ensure mandatory objects maintain their required spec.
- Objects are immutable; nobody should modify the returned maps.
