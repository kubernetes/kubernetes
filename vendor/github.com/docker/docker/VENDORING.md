# Vendoring policies

This document outlines recommended Vendoring policies for Docker repositories.
(Example, libnetwork is a Docker repo and logrus is not.)

## Vendoring using tags

Commit ID based vendoring provides little/no information about the updates
vendored. To fix this, vendors will now require that repositories use annotated
tags along with commit ids to snapshot commits. Annotated tags by themselves
are not sufficient, since the same tag can be force updated to reference
different commits.

Each tag should:
- Follow Semantic Versioning rules (refer to section on "Semantic Versioning")
- Have a corresponding entry in the change tracking document.

Each repo should:
- Have a change tracking document between tags/releases. Ex: CHANGELOG.md,
github releases file.

The goal here is for consuming repos to be able to use the tag version and
changelog updates to determine whether the vendoring will cause any breaking or
backward incompatible changes. This also means that repos can specify having
dependency on a package of a specific version or greater up to the next major
release, without encountering breaking changes.

## Semantic Versioning
Annotated version tags should follow [Semantic Versioning](http://semver.org) policies:

"Given a version number MAJOR.MINOR.PATCH, increment the:

   1. MAJOR version when you make incompatible API changes,
   2. MINOR version when you add functionality in a backwards-compatible manner, and
   3. PATCH version when you make backwards-compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions
to the MAJOR.MINOR.PATCH format."

## Vendoring cadence
In order to avoid huge vendoring changes, it is recommended to have a regular
cadence for vendoring updates. e.g. monthly.

## Pre-merge vendoring tests
All related repos will be vendored into docker/docker.
CI on docker/docker should catch any breaking changes involving multiple repos.
