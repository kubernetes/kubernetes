# Package: events

## Purpose
Internal (unversioned) API types for the events.k8s.io API group. Reuses core.Event and core.EventList types for the events API group.

## Key Functions

- **Kind(kind string)**: Returns Group-qualified GroupKind.
- **Resource(resource string)**: Returns Group-qualified GroupResource.
- **AddToScheme**: Registers Event and EventList types.

## Key Constants

- **GroupName**: "events.k8s.io"

## Design Notes

- Reuses core.Event type rather than defining new types.
- The events.k8s.io API group provides a more scalable events API than core/v1 Events.
- Supports event aggregation and deduplication.
