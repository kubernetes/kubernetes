# Package: storage

## Purpose
Provides REST storage implementation for VolumeAttributesClass objects.

## Key Types

- **REST**: Wraps genericregistry.Store for VolumeAttributesClass.

## Key Functions

- **NewREST(optsGetter)**: Creates REST storage for VolumeAttributesClass:
  - Uses volumeattributesclass.Strategy for all operations
  - Returns deleted object on delete
  - Includes TableConvertor for kubectl output formatting

- **ShortNames()**: Returns ["vac"] - the short name for VolumeAttributesClass.
