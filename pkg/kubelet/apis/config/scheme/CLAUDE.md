# Package: scheme

## Purpose
This package provides utility functions for working with the kubelet configuration API scheme. It creates runtime Scheme and CodecFactory instances that understand all versions of the kubeletconfig API types.

## Key Functions

- **NewSchemeAndCodecs()**: Returns a Scheme and CodecFactory for kubeletconfig types

## Supported Versions

Registers types from:
- Internal kubeletconfig types
- kubeletconfig/v1 (GA version)
- kubeletconfig/v1beta1 (Beta version)

## Design Notes

- Accepts CodecFactoryOptionsMutator for customizing codec behavior (e.g., strict decoding)
- Returns error if scheme registration fails
- Used by kubelet startup code to decode configuration files
- Enables conversion between different API versions
