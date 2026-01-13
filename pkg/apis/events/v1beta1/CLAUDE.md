# Package: v1beta1

## Purpose
Provides conversion and defaulting logic for events.k8s.io/v1beta1 API types (Event).

## Key Functions

- **Resource(resource string)**: Returns Group-qualified GroupResource for v1beta1.
- **AddToScheme**: Registers v1beta1 types with conversion and field label functions.
- **RegisterDefaults**: Registers defaulting functions.
- **AddFieldLabelConversionsForEvent**: Registers field label selectors for Events.

## Key Constants

- **GroupName**: "events.k8s.io"
- **SchemeGroupVersion**: events.k8s.io/v1beta1

## Design Notes

- Beta version, maintained for backward compatibility.
- External types defined in k8s.io/api/events/v1beta1.
