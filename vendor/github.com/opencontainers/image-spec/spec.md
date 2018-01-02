# Open Container Initiative
## Image Format Specification

This specification defines an OCI Image, consisting of a [manifest](manifest.md), an [image index](image-index.md) (optional), a set of [filesystem layers](layer.md), and a [configuration](config.md).

The goal of this specification is to enable the creation of interoperable tools for building, transporting, and preparing a container image to run.

### Table of Contents

- [Introduction](spec.md)
- [Notational Conventions](#notational-conventions)
- [Overview](#overview)
    - [Understanding the Specification](#understanding-the-specification)
    - [Media Types](media-types.md)
- [Content Descriptors](descriptor.md)
- [Image Layout](image-layout.md)
- [Image Manifest](manifest.md)
- [Image Index](image-index.md)
- [Filesystem Layers](layer.md)
- [Image Configuration](config.md)
- [Annotations](annotations.md)
- [Considerations](considerations.md)
    - [Extensibility](considerations.md#extensibility)
    - [Canonicalization](considerations.md#canonicalization)

## Notational Conventions

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" are to be interpreted as described in [RFC 2119](http://tools.ietf.org/html/rfc2119) (Bradner, S., "Key words for use in RFCs to Indicate Requirement Levels", BCP 14, RFC 2119, March 1997).

The key words "unspecified", "undefined", and "implementation-defined" are to be interpreted as described in the [rationale for the C99 standard][c99-unspecified].

An implementation is not compliant if it fails to satisfy one or more of the MUST, MUST NOT, REQUIRED, SHALL, or SHALL NOT requirements for the protocols it implements.
An implementation is compliant if it satisfies all the MUST, MUST NOT, REQUIRED, SHALL, and SHALL NOT requirements for the protocols it implements.

## Overview

At a high level the image manifest contains metadata about the contents and dependencies of the image including the content-addressable identity of one or more [filesystem layer changeset](layer.md) archives that will be unpacked to make up the final runnable filesystem.
The image configuration includes information such as application arguments, environments, etc.
The image index is a higher-level manifest which points to a list of manifests and descriptors.
Typically, these manifests may provide different implementations of the image, possibly varying by platform or other attributes.

![](img/build-diagram.png)

Once built the OCI Image can then be discovered by name, downloaded, verified by hash, trusted through a signature, and unpacked into an [OCI Runtime Bundle](https://github.com/opencontainers/runtime-spec/blob/master/bundle.md).

![](img/run-diagram.png)

### Understanding the Specification

The [OCI Image Media Types](media-types.md) document is a starting point to understanding the overall structure of the specification.

The high-level components of the spec include:

* [Image Manifest](manifest.md) - a document describing the components that make up a container image
* [Image Index](image-index.md) - an annotated index of image manifests
* [Image Layout](image-layout.md) - a filesystem layout representing the contents of an image
* [Filesystem Layer](layer.md) - a changeset that describes a container's filesystem
* [Image Configuration](config.md) - a document determining layer ordering and configuration of the image suitable for translation into a [runtime bundle][runtime-spec]
* [Descriptor](descriptor.md) - a reference that describes the type, metadata and content address of referenced content

Future versions of this specification may include the following OPTIONAL features:

* Signatures that are based on signing image content address
* Naming that is federated based on DNS and can be delegated

[c99-unspecified]: http://www.open-std.org/jtc1/sc22/wg14/www/C99RationaleV5.10.pdf#page=18
[runtime-spec]: https://github.com/opencontainers/runtime-spec
