# libcontainer: what's next?

This document is a high-level overview of where we want to take libcontainer next.
It is a curated selection of planned improvements which are either important, difficult, or both.

For a more complete view of planned and requested improvements, see [the Github issues](https://github.com/docker/libcontainer/issues).

To suggest changes to the roadmap, including additions, please write the change as if it were already in effect, and make a pull request.

## Broader kernel support

Our goal is to make libcontainer run everywhere, but currently libcontainer requires Linux version 3.8 or higher. If you’re deploying new machines for the purpose of running libcontainer, this is a fairly easy requirement to meet. However, if you’re adding libcontainer to an existing deployment, you may not have the flexibility to update and patch the kernel.

## Cross-architecture support

Our goal is to make libcontainer run everywhere. Recently libcontainer has
expanded from its initial support for x86_64 systems to include POWER (ppc64
little and big endian variants), IBM System z (s390x 64-bit), and ARM. We plan
to continue expanding architecture support such that libcontainer containers
can be created and used on more architectures.
