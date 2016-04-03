# Hacking on Docker

The hack/ directory holds information and tools for everyone involved in the process of creating and
distributing Docker, specifically:

## Guides

If you're a *contributor* or aspiring contributor, you should read CONTRIBUTORS.md.

If you're a *maintainer* or aspiring maintainer, you should read MAINTAINERS.md.

If you're a *packager* or aspiring packager, you should read PACKAGERS.md.

If you're a maintainer in charge of a *release*, you should read RELEASE-CHECKLIST.md.

## Roadmap

A high-level roadmap is available at [ROADMAP.md](../ROADMAP.md).


## Build tools

make.sh is the primary build tool for docker. It is used for compiling the official binary,
running the test suite, and pushing releases.
