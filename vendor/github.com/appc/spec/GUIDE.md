# Implementation Guide

This document contains guidelines, tips and suggested best practices for implementers of the App Container specification.

It should not be considered prescriptive, but is instead a general resource to guide usability and optimization of implementations.

This is a living document; those implementing the spec are encouraged to submit feedback and improvements.

## Storing and working with App Container Images (ACIs)

The canonical ACI format is a simple tarball, which conveniently encapsulates the filesystem and metadata of an image in a single file.
This also provides a simple, unique and canonical way to reference an image, using the cryptographic hash of this artefact.
Very simple implementations of the spec might use this tarball format as their primary format for transport and local storage, extracting the tarball as necessary every time an image needs to be accessed.
However, when looking to minimize transport and startup times for running applications, this is inefficient; a variety of optimizations are possible, both for servers hosting images ("image repositories") and for local stores (for example any caches internal to Application Container Executors)

**Note that since images retaining a cryptographic identity is a key part of the specification, it should always be possible to recreate the canonical format of an image (and hence its cryptographic Image ID), regardless of what optimizations are leveraged.**

### Indexing ImageManifests

A simple example would be indexing the ImageManifest outside of the ACI.
For example, an image repository might use an RDBMS and store the information contained in ImageManifests in a normalized fashion.
This would facilitate efficient lookups of images based on certain attributes - for example, querying based on a combination of name and labels.

### Storing exploded ACIs

A similar optimization relates to the storage on disk of the root filesystem.
Instead of hosting the tarball form of an image on disk, and expanding it every time it is referenced, stores could simply store the root filesystem as an expanded tree on disk.
When taking this approach, care must be taken to preserve all the appropriate metadata that the tarball format encapsulates.
Storing exploded ACIs not only prevents the need to regularly tar and untar images (and hence improves performance for launching containers), but it also allows for further space optimizations: for example, if files are stored in a content-addressable-store, then they can be de-duplicated between ACIs.
(Future versions of the spec will likely define a streaming image format which would facilitate similar behaviour.)

### De-duplicating images at runtime

The specification stipulates that each execution of an application must start from a clean copy of its image.
The most basic implementation of this would involve a complete copy (e.g. `cp -arv`) of the image's root filesystem into a new directory for the application to use.
To make more efficient use of disk space, and improve performance, implementations should consider use of technologies like [device mapper copy-on-write block devices] [dm], [OverlayFS] [overlay], or [ZFS] [zfs].

[dm]: https://www.kernel.org/doc/Documentation/device-mapper/snapshot.txt
[overlay]: https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/tree/Documentation/filesystems/overlayfs.txt
[zfs]: http://en.wikipedia.org/wiki/ZFS

## Transporting ACIs

### Image Discovery

While the specification prescribes that the Meta Discovery process occurs over HTTP(S), it is intentionally agnostic with respect to the transport over which the discovered resource can be retrieved.
In the simplest cases, the ACI payload can simply be retrieved over HTTPS itself.
However, in more advanced implementations - particularly in highly distributed environments - alternative protocols like [HDFS][hdfs] or [BitTorrent][bittorrent] could be used instead.

[hdfs]: http://hadoop.apache.org/docs/r1.2.1/hdfs_design.html
[bittorrent]: http://en.wikipedia.org/wiki/BitTorrent
