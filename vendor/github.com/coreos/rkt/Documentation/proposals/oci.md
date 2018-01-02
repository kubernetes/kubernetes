## OCI Image Format roadmap

### Background

rkt currently implements the [appc specification][app-container] and therefore relies on the [ACI][aci] (Application Container Image) image format internally.

[OCI][opencontainers] on the other hand defines a new [image format][oci] following a separate specification.
This new specification differs considerably from rkt's internal ACI-based image format handling.

The internal rkt image handling is currently divided in three subsystems:
- **fetching**: This subsystem is responsible for downloading images of various types.
Non-ACI image types (Docker and OCI) are converted to ACI images by delegating to [docker2aci][docker2aci]. The logic resides in the `github.com/coreos/rkt/rkt/image` package.

- **image store**: The image store is responsible for persisting and managing downloaded images.
It consists of two parts, a directory tree storing the actual image file blobs (usually residing under `/var/lib/rkt/cas/blob`) and a separate embedded SQL database storing image metadata usually residing in `/var/lib/rkt/cas/db/ql.db`. The implementation resides in the `github.com/coreos/rkt/store/imagestore` package.

- **tree store**: Since dependencies between ACI images form a directed acyclic graph according to the [appc spec][ace-fs] they are pre-rendered in a directory called the tree store cache.
If the [overlay filesystem](https://www.kernel.org/doc/Documentation/filesystems/overlayfs.txt) is enabled, the pre-rendered image is used as the `lowerdir` for the pod's rendered rootfs. The implementation resides in the `github.com/coreos/rkt/store/treestore` package.

The actual internal lifecycle of an image is documented in the [architecture documentation][image-lifecycle].

The following table gives an overview of the relevant differences between OCI and appc regarding image handling aspects:

 Aspect | OCI | ACI
--------|-----|------------------
Dependencies | Layers array in the [image manifest][oci-manifest] | [Dependency graph][ace-fs]
Hash algorithms | Potentially multiple [algorithms][oci-algorithms], but SHA-256 preferred | [SHA-512][appc-image-id-type]

Current ongoing work to support OCI in rkt is tracked in the following Github project: [https://github.com/coreos/rkt/projects/4](https://github.com/coreos/rkt/projects/4).

### Goals, non-Goals

With the deprecation of the appc Spec (https://github.com/appc/spec#-disclaimer-) the current internal rkt architecture is not favorable any more.
Currently rkt does support ACI, Docker, and OCI images, but the conversion step from OCI to ACI using `docker2aci` seems unnecessary.
It introduces CPU and I/O bound overhead and is bound by semantical differences between the formats. For these reasons native support of OCI images inside rkt is envisioned.

The goal therefore is to support OCI images natively next to ACI.

This document outlines the following necessary steps and references existing work to transition to native OCI image support in rkt:

1. Distribution points
2. Reference based image handling
3. Transport handlers
4. Tree store support for OCI

A non-goal is the implementation of the [OCI runtime specification][oci-runtime]. There is ongoing work in [https://github.com/coreos/rkt/issues/3408](https://github.com/coreos/rkt/issues/3408) covering this topic.

### Overview

#### Distribution points

rkt historically used the image name and heuristics around it to determine the actual image format type (appc, Docker, OCI).
The concept of "distribution points" introduced a URI syntax that uniquely points to the different image formats including the necessary metadata (file location, origin URL, version, etc.).

The URI scheme "cimd" (Container Image Distribution) was chosen to uniquely identify different image formats. The following CIMDs are currently supported:

Name | Example
-----|--------
appc | `cimd:appc:v=0:coreos.com/etcd?version=v3.0.3&os=linux&arch=amd64`
ACIArchive | `cimd:aci-archive:v=0:file%3A%2F%2Fabsolute%2Fpath%2Fto%2Ffile`
Docker | `cimd:docker:v=0:busybox`

The design document can be found in [Documentation/devel/distribution-point.md][distribution-point].

###### Status

- The design document (https://github.com/coreos/rkt/pull/2953) is merged.
- The implementation (https://github.com/coreos/rkt/pull/3369) is merged.

###### TODOs

- Introduce a dedicated remote `cimd:oci` and potentially also a local `cimd:oci-layout` (see [github.com/opencontainers/image-spec/image-layout.md][oci-image-layout]) CIMD scheme.

#### Reference based image handling

The current image store implementation does not support different image formats. The blob image store only supports SHA-512.
The ql backed SQL image store has a simple SQL scheme referencing only ACI images.

In order to prepare native support for OCI the following changes need to be implemented:

- Store the CIMD URI as a primary key in the current image store.
- Support for multiple hash algorithms: Currently only SHA-512 is supported. OCI in addition needs SHA-256 and potentially other hash algorithms.
- The database schema needs to be reworked to reflect multi-image support.

###### Status

- The design and initial implementation is proposed in https://github.com/coreos/rkt/pull/3071.
- The actual design document of the above PR can be found in [Documentation/proposals/reference_based_image_access_and_cas_store.md](https://github.com/coreos/rkt/blob/23313af1c3dac2fb24fe41f9a7c5eaca573e45dd/Documentation/proposals/reference_based_image_access_and_cas_store.md).

Note that the above design document also suggests the introduction of a new key/value [Bolt](https://github.com/boltdb/bolt) based store. The current consensus is that the replacement of `ql` as the backing store can be done independently and therefore should be a non-goal for the OCI roadmap.

###### TODOs

- Finalize the initial design proposal and implementation.

#### Transport handlers

Currently rkt directly fetches remote ACI based images or uses `docker2aci` to delegate non-ACI fetching.
The current implementation makes it hard to integrate separate fetching subsystems due to the lack of any abstraction.

###### Status

The current proposal is to abstract fetching logic behind "transport handlers" allowing for independent (potentially swappable) fetching implementations for the various image formats.

- A first initial design is proposed in https://github.com/coreos/rkt/pull/2964.
- The actual design document of the above PR can be found in [Documentation/proposal/fetchers_refactor.md](https://github.com/sgotti/rkt/blob/239fdff081f9fd47dd08834a5660a1375ea4771d/Documentation/proposal/fetchers_refactor.md).
- A first initial implementation is proposed in https://github.com/coreos/rkt/pull/3232.

Note that the initial design and implementation are in very early stage and should only be considered inspirational. 

###### TODOs

- Fetching images remotely and locally from disk for all formats must be supported.
- The current initial design proposal needs to be finalized.
- The current fetcher logic needs to be abstracted allowing to introduce alternative libraries like https://github.com/containers/image to delegate fetching logic for OCI or Docker images.

#### Tree store support for OCI

The current tree store implementation is used for rendering ACI images only. A design document and initial implementation has to be created to prototype deflating OCI images and their dependencies.

###### Status

Not started yet

### Risks

Backwards compatibility: Currently the biggest concern identified is backwards compatibility/rollback capabilities. The proposed changes do not only imply simple schema changes in the `ql` backed database, but also intrusive schema and directory layout changes.

[opencontainers]: https://www.opencontainers.org/
[docker2aci]: https://github.com/appc/docker2aci
[oci-runtime]: https://github.com/opencontainers/runtime-spec

[aci]: https://github.com/appc/spec/blob/v0.8.10/spec/aci.md#app-container-image
[ace-fs]: https://github.com/appc/spec/blob/v0.8.10/spec/ace.md#filesystem-setup
[appc-image-id-type]: https://github.com/appc/spec/blob/v0.8.10/spec/types.md#image-id-type

[oci]: https://github.com/opencontainers/image-spec
[oci-manifest]: https://github.com/opencontainers/image-spec/blob/v1.0.0-rc2/manifest.md#image-manifest
[oci-algorithms]: https://github.com/opencontainers/image-spec/blob/v1.0.0-rc2/descriptor.md#algorithms
[oci-image-layout]: https://github.com/opencontainers/image-spec/blob/v1.0.0-rc2/image-layout.md

[app-container]: https://github.com/coreos/rkt/blob/v1.25.0/Documentation/app-container.md
[image-lifecycle]: https://github.com/coreos/rkt/blob/v1.25.0/Documentation/devel/architecture.md#image-lifecycle
[distribution-point]: https://github.com/coreos/rkt/blob/v1.25.0/Documentation/devel/distribution-point.md
