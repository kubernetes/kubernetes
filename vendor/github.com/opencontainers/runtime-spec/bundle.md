# <a name="filesystemBundle" />Filesystem Bundle

## <a name="containerFormat" />Container Format

This section defines a format for encoding a container as a *filesystem bundle* - a set of files organized in a certain way, and containing all the necessary data and metadata for any compliant runtime to perform all standard operations against it.
See also [MacOS application bundles][macos_bundle] for a similar use of the term *bundle*.

The definition of a bundle is only concerned with how a container, and its configuration data, are stored on a local filesystem so that it can be consumed by a compliant runtime.

A Standard Container bundle contains all the information needed to load and run a container.
This includes the following artifacts:

1. <a name="containerFormat01" />`config.json`: contains configuration data.
    This REQUIRED file MUST reside in the root of the bundle directory and MUST be named `config.json`.
    See [`config.json`](config.md) for more details.

2. <a name="containerFormat02" />container's root filesystem: the directory referenced by [`root.path`](config.md#root), if that property is set in `config.json`.

When supplied, while these artifacts MUST all be present in a single directory on the local filesystem, that directory itself is not part of the bundle.
In other words, a tar archive of a *bundle* will have these artifacts at the root of the archive, not nested within a top-level directory.

[macos_bundle]: https://en.wikipedia.org/wiki/Bundle_%28macOS%29
