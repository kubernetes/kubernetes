# Package: fieldpath

## Purpose
This package provides utilities for extracting field values from Kubernetes API objects using field path expressions. It is primarily used for the Downward API to expose pod and container metadata to running containers.

## Key Functions

- **ExtractFieldPathAsString()**: Extracts a field value from an object given a field path expression
- **FormatMap()**: Formats a map[string]string as a sorted key=value string (for labels/annotations)
- **SplitMaybeSubscriptedPath()**: Parses subscripted field paths like "metadata.annotations['key']"

## Supported Field Paths

- `metadata.name`: Pod name
- `metadata.namespace`: Pod namespace
- `metadata.uid`: Pod UID
- `metadata.labels`: All labels as formatted string
- `metadata.annotations`: All annotations as formatted string
- `metadata.labels['key']`: Specific label value
- `metadata.annotations['key']`: Specific annotation value

## Design Notes

- Uses meta.Accessor to extract metadata from any API object
- Validates subscript keys against Kubernetes naming rules
- Output is deterministic with sorted keys for map formatting
- Subscript syntax uses single quotes: path['key']
- Annotation keys are validated case-insensitively, label keys case-sensitively
