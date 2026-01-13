# Package: image

## Purpose
Implements the Image volume plugin (stub) for KEP-4639 that will allow container images to be mounted as read-only volumes.

## Key Types/Structs
- `imagePlugin` - VolumePlugin for image volumes

## Key Functions
- `ProbeVolumePlugins()` - Returns the image volume plugin
- `CanSupport()` - Checks if volume spec has image source
- `NewMounter()` - Returns nil (not implemented yet)

## Design Patterns
- Currently a stub/placeholder implementation
- Part of KEP-4639 ImageVolume feature
- Will enable mounting OCI images as volumes
- Feature is under development and not yet functional
