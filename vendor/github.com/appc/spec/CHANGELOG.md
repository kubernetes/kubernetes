### v0.7.4

Minor release of the spec with some enhancements to the schema types and discovery code:
- Added `AsIsolator` constructors for Memory and CPU resource isolators (#552)
- Added insecure flags for HTTP and TLS to discovery code, for more granular control over security during discovery (#545, #551)

### v0.7.3

This is a minor release of the spec with one bug fix over the previous release:
- Fixed the `AsIsolator` function on the LinuxCapabilitiesSet isolator types so it correctly populates the value

### v0.7.2

This is a minor release of the spec which should be fully backwards-compatible but extends functionality in several ways and loosens some restrictions.

Spec changes:
- The requirement for the metadata service has been downgraded to a SHOULD. This requirement necessitated a daemon which seemed burdensome for implementations.
- Added a SHOULD requirement for Linux ACEs to provide a basic `/etc/hosts` if none already exists in application filesystems
- The requirement for `exec` fields in the `app` schema to be non-empty was removed. An ACE is permitted to override/replace this section and so this provides greater flexibility when generating images, particularly when converting from other image types to ACI.
- Added `mode`, `uid`, and `gid` parameters to empty volumes. This allows setting which permissions are applied to empty volumes. 
- Changed the definition of the executable path in `app.exec` to be PATH-dependent. The procedure for an ACE to locate the executable path mimics that of the shell (as described in `man 3 exec`). This means that executable paths are no longer required to be absolute.

Other changes:
- Improved manifest parsing errors - when provided manifests are invalid JSON, the erroneous line and column number will now be highlighted in the produced error
- The discovery code now includes per-host HTTP headers, allowing authentication during discovery
- Added several new helper functions for initializing memory and CPU isolator types
- Added several helpers to work with LinuxCapabilitiesSet schema types
- Refactored the MakeQueryString helper to centralise the parsing of different comma-separated label/value strings used in several places (for volumes, ports, mountpoints, etc). This also now escapes values when parsing, allowing values with special URL characters like "+" or "&"
- Fixed a nil-pointer dereference in the schema `Volume` type's `String()` method

### v0.7.1

Minor release of the spec with one critical bug/consistency fix and a few tooling enhancements.

0.7.0 introduced a field to the app section of ImageManifests to allow users to specify supplementary group IDs. Unfortunately, this was implemented with inconsistent naming: `supplementaryGids` in the text of the spec itself, but `supplementaryGroups` in the schema code.

In this release we standardise both the spec and the schema to `supplementaryGIDs`. See #516 for more information.

Other changes in this release:
- Added a callback to BuildWalker in the aci package to allow users to modify tar entries while building an ACI (#509)
- Added an `--owner-root` flag to acbuild to adjust the uid/gid of all files in an ACI to 0:0 (#509)
- Added a `--supplementary-gids` flag to actool's patch-manifest subcommand to adjust the supplementary group IDs of an ACI (#506, #516)
- Added the ability to extract labels in the `lastditch` package (#508)
- Changed the behaviour of the `NewAppFromString` parser in the `discovery` package to URL-encode label values before parsing them (#514)

### v0.7.0

Next major release of the spec, with a lot of tooling improvements, wording clarifications, and one breaking schema change from the previous release.

Spec changes since v0.6.1:
- The `mount` objects in pod manifests now refer directly to a path rather than referencing an app's mountPoint. This makes it easier for implementations to provide supplementary, arbitrary mounts in pods without needing to modify app sections in the pod manifest. This is a breaking schema change; however, implementations are suggested to preserve similar behaviour to the previous three-level mount mappings by continuing to use mountPoints name fields to generate mount<->volume mappings (#364, #495)
- The resource/cpu unit has changed to cpu cores, instead of millicores (#468)
- The wording around unpacking of ACI dependencies was reworked to clarify the expected order of unpacking (#425, #431, #494)
- The wording around mounting volumes was expanded to advise much more explicit behaviour and protect various attack vectors for hosts (#431, #494)
- App sections now support a `supplementaryGroups` field, which allows users to specify a list of additional GIDs that the processes of the app should run with (#339)
- A new "Dependency Matching" section better explains the label matching process during dependency resolution (#469)
- Clarified wording around the "empty" volume type (#449)

Tooling changes and features:
- Added an `IsValidOSArch` function implementations can use
- When patching ACIs that do not have an `app` section in their manifest, actool will now automatically inject an App iff the user specifies an exec statement (#473, #489)
- actool now warns if a manifest's ACVersion is too old (#322)
- The NewCompressedReader function in the aci package now returns an io.ReadCloser instead of an io.Reader to facilitate closing the reader in the case of xz compression. (#462)
- Added a new last-ditch parser of pod and image manifests to facilitate retrieving some debugging information for badly-formed manifests (#477)

Schema and tooling bugfixes:
- actool now validates layouts before opening output file so that it does not create empty files or truncate existing ones if a layout is invalid (#322)
- Fixed a panic when `actool patch-manifest` is supplied with an ACI with no app section in its ImageManifest (#473)
- ACE validator's name is ACName compatible (#448)
- ACE validator's mountpoint checking is now more robust on Linux and works on FreeBSD (#467)
- `build_aci` now works if NO_SIGNATURE is set
- `build_aci` properly generates an armored signature (#460)
- Fixed a typo in the ACE validator (uid should be uuid) (#485)

Other changes:
- Rewrote the kubernetes import path in Godeps (#471)
- pkg/device is now buildable on OS X (#486)
- schema code is now tested against Go 1.5
- Added explicit reference to RFC2119 regarding wording

### v0.6.1

Minor release of the spec; the most important change is adjusting the type for
`annotations` to ACIdentifier (#442). This restores the ability for annotation
fields to be namespaced with DNS names.

Other changes:
- Added new maintainer (Ken Robertson)
- Created CHANGELOG.md to track changes instead of using git tags
- Fixed build scripts for FreeBSD (#433)
- Fixed acirenderer to work properly with empty images with just a rootfs
  directory (#428)
- Added `arm6vl` as valid arch for Linux (#440)

### v0.6.0

This is an important milestone release of the spec. Critically, there are two
backwards-incompatible schema changes from previous releases:
- Dependency references' `app` field has been renamed to the more accurate and
  unambiguous `imageName`: #397
- `ACName` has been redefined (with a stricter specification) to be suitable
  for relative names within a pod, and a new type, `ACIdentifier`, has been
  introduced for image, label and isolator names: #398

This release also sees the sections of the specification - image format,
discovery process, pods and executor definitions - being split into distinct
files. This is a first step towards clarifying the componentised nature of the
spec and the value in implementing individual sections.

Other changes of note in this release:
- Dependency references gained an optional `size` field. If this field is
  specified, executors should validate that the size of resolved dependency
  images is correct before attempting to retrieve them: #422
- The RFC3339 timestamp type definition was tweaked to clarify that it must
  include a T rather than a space: #410
- The spec now prescribes that ACEs must set the `container` environment
  variable to some value to indicate to applications that they are being run
  inside a container: #302
- Added support for 64-bit big-endian ARM architectures: #414
- Clarifications to the ports definition in the schema: #405
- Fixed a bug in the discovery code where it was mutating supplied objects:
  #412

### v0.5.2

This release features a considerable number of changes over the previous
(0.5.1) release. However, the vast majority are syntactical improvements to
clarity and wording in the text of the specification and do not change the
semantic behaviour in any significant way; hence, this should remain a
backwards-compatible release. As well as the changes to the spec itself, there
are various improvements to the schema/tooling code, including new
functionality in `actool`.

Some of the more notable changes since v0.5.1:
- `linux/aarch64`, `linux/armv7l` and `linux/armv7b` added as recognised
  os/arch combinations for images
- added contribution/governance policy and new maintainers
- added `cat-manifest` and `patch-manifest` subcommands to actool to manipulate
  existing ACIs
- added guidance around using authorization token (supplied in AC_METADATA_URL)
  for identifying pods to the metadata service
- reduced the set of required environment variables that executors must provide
- fixed consistency between schema code and spec for capabilities
- all TODOs removed from spec text and moved to GitHub issues
- several optimizations and fixes in acirenderer package

### v0.5.1

This is primarily a bugfix release to catch 2a342dac which resolves an issue
preventing PodManifests from being successfully serialized.

Other changes:
- Update validator to latest Pod spec changes
- Added /dev/pts and /dev/ptmx to Linux requirements
- Added a script to bump release versions
- Moved to using types.ACName in discovery code instead of strings

### v0.5.0

The major change in this release is the introduction of _pods_,
via #207 and #248. Pods are a refinement (and replacement) of the
previous ContainerRuntimeManifest concept that define the minimum
deployable, executable unit in the spec, as a grouping of one or
more applications. The move to pods implicitly resolves various
issues with the CRM in previous versions of the spec (e.g #83, #84)

Other fixes and changes in this release:
- fix static builds of the tooling on go1.4
- add ability to use proxy from environment for discovery
- fix inheritance of readOnly flag
- properly validate layouts with relative paths
- properly tar named pipes and ignore sockets
- add /dev/shm to required Linux environment

### v0.4.1

This is a minor bugfix release to fix marshalling of isolators.

### v0.4.0

Major changes and additions since v0.3.0:
- Reworked isolators to objects instead of strings and clarify limits vs
  reservations for resource isolators
- Introduced OS-specific requirements (e.g. device files on Linux)
- Moved much of the wording in the spec towards RFC2119 wording ("MUST", "MAY",
  "SHOULD", etc) to be more explicit about which parts are
  required/optional/recommended
- Greater explicitness around signing and encryption requirements
- Moved towards `.asc` filename extension for signatures
- Added MAINTAINERS
- Added an implementation guide
- Tighter restrictions on layout of ACI tars
- Numerous enhancements to discovery code
- Improved test coverage for various schema types

### v0.3.0

### v0.2.0

### v0.1.1

This marks the first versioned release of the app container spec.
