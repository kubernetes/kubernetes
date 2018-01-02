# Build Configuration

## Overview

This file describes all the parameters of the configure script and their possible uses.
For a quick help of available parameters, run `./configure --help`.

## Parameters for building stage1 flavors

#### `--with-stage1-flavors`

This parameter takes a comma-separated list of all the flavors that the build system should assemble.
Depending on a default stage1 image setup, this list is by default either empty or set to `coreos,kvm,fly` for, respectively, detailed setup and flavor setup.
Note that specifying this parameter does not necessarily mean that rkt will use them in the end.
Available flavors are:

- `coreos` - it takes systemd and bash from a CoreOS PXE image; uses systemd-nspawn
- `kvm` - it takes systemd, bash and other binaries from a CoreOS PXE image; uses lkvm or qemu
- `src` - it builds systemd, takes bash from the host at build time; uses built systemd-nspawn
- `host` - it takes systemd and bash from host at runtime; uses systemd-nspawn from the host
- `fly` - chroot-only approach for single-application minimal isolation containers; native Go implementation

The `host` flavor is probably the best suited flavor for distributions that have strict rules about software sources.

#### `--with-stage1-flavors-version-override`

This parameter takes a version number to become the version of all the built stage1 flavors.
Normally, without this parameter, the images have the same version as rkt itself.
This parameter may be useful for distributions that often provide patched versions of upstream software without changing major/minor/patch version number, but instead add a numeric suffix.
An example usage could be passing `--with-stage1-flavors-version-override=0.12.0-2`, so the new image will have a version `0.12.0-2` instead of `0.12.0`.
This parameter also affects the default stage1 image version in flavor setup.

## Parameters for setting up default stage1 image

The parameters described below affect the handling of rkt's default stage1 image.
rkt first tries to find the stage1 image in the store by using the default stage1 image name and version.
If this fails, rkt will try to fetch the image into the store from the default stage1 image location.

There are two mutually exclusive ways to specify a default stage1 image name and version:

- flavor setup
- detailed setup

### Flavor setup

Flavor setup has only one parameter.
This kind of setup is rather a convenience wrapper around the detailed setup.

#### `--with-stage1-default-flavor`

It takes a name of the flavor of the stage1 image we build and, based on that, it sets up the default stage1 image name and version.
Default stage1 image in this case is often something like coreos.com/rkt/stage1-<name of the flavor>.
Default stage1 version is usually just rkt version, unless it is overridden with `--with-stage1-flavors-version-override`.
This is the default setup if neither flavor nor detailed setup are used.
The default stage1 image flavor is the first flavor on the list in `--with-stage1-flavors`.

### Detailed setup

Detailed setup has two parameters, both must be provided.
This kind of setup could be used to make some 3rd party stage1 implementation the default stage1 image used by rkt.

#### `--with-stage1-default-name`

This parameter tells what is the name of the default stage1 image.

#### `--with-stage1-default-version`

This parameter tells what is the version of the default stage1 image.

### Location of the default stage1 image

#### `--with-stage1-default-location`

This parameter tells rkt where to find the default stage1 image if it is not found in the store.
For the detailed setup, the default value of this parameter is empty, so if it is not provided, you may be forced to inform rkt about the location of the stage1 image at runtime.
For the flavor setup, the default value is also empty, which tells rkt to look for the image in the directory the rkt binary is located, unless it is overridden at runtime.
Normally, this parameter should be some URL, with a scheme or an absolute path.

#### `--with-stage1-default-images-directory`

This parameter tells rkt where the directory which contains all the stage1 images is located.
The value should be an absolute path.
In this directory, all the built flavors of stage1 images should be installed.
The `--stage1-from-dir` rkt flag will look for images in this directory.
The default value of this parameter is `<libdir>/rkt/stage1-images`, where `<libdir>` is a distribution-specific place for storing arch-dependent files.

## Flavor-specific parameters

There are some additional parameters for some flavors.
Usually they do not need to be modified, default values are sane.

### `src` flavor

`src` flavor provides parameters for specifying some `git`-specific details of the systemd repository.

#### `--with-stage1-systemd-src`

This parameter takes a URL to a `systemd` git repository.
The default is `https://github.com/systemd/systemd.git`.
You may want to change it to point the build system to use some local repository.

#### `--with-stage1-systemd-version`

This parameter specifies the systemd version to be built.
Version names are usually in form of `v<number>`, where number is a systemd version.
The default is `v999`.

#### `--with-stage1-systemd-revision`

This parameter takes either a tag name or a branch name.
You can use branch name `master` to test the bleeding edge version of systemd or any working branch, or tag name.
Since arbitrary branch names do not imply which systemd version is being built, the actual systemd version
is specified using `--with-stage1-systemd-version`.
The default is `master`.

### `coreos` and `kvm` flavor

`coreos` and `kvm` flavors provide parameters related to CoreOS PXE image.

#### `--with-coreos-local-pxe-image-path`

This parameter is used to point the build system to a local CoreOS PXE image.
This can be helpful for some packagers, where downloading anything over the network is a no-no.
The parameter takes either relative or absolute paths.
The default value is empty, so the image will be downloaded over the network.
If this parameter is specified, then also `--with-coreos-local-pxe-image-systemd-version` must be specified too.

#### `--with-coreos-local-pxe-image-systemd-version`

The build system has no reliable way to deduce automatically what version of systemd the CoreOS PXE image contains, so it needs some help.
This parameters tells the build systemd what is the version of systemd in the local PXE image.
The value should be like tag name in systemd git repository, that is - `v<number>`, like `v229`.
If this parameter is specified, then also `--with-coreos-local-pxe-image-path` must be specified too.

## Testing

There is only one flag for testing - to enable functional testing.

#### `--enable-functional-tests`

Functional tests are disabled by default.
There are some requirements to be fulfilled to be able to run them.
The tests are runnable only in Linux.
The tests must be run as root, so the build system uses sudo to achieve that.
Note that using sudo may kill the non-interactivity of the build system, so make sure that if you use it in some CI, then CI user is a sudoer and does not need a password.
Also, when trying to run functional tests with the host flavor of the stage1 image, the host must be managed by systemd of at least version v220.
If any of the requirements above are not met and the value of the parameter is yes then configure will bail out.
This may not be ideal in CI environment, so there is a third possible value of this parameter - "auto".
"auto" will enable functional tests if all the requirements are met.
Otherwise, it will disable them without any errors.

## Security

These flags are related to security.

#### `--enable-tpm`

This option to enable [logging to the TPM][rkt-tpm] is set by default. For logging to work, [TrouSerS][trousers] is required. Set this option to `auto` to conditionally enable TPM features based on build support.

#### `--enable-insecure-go`

This option to allow building rkt with go having known security issues is unset by default. Use it with caution.

## Development

#### `--enable-incremental-build`

This option enables incremental compilation. This is useful for local development.
In contrast to a release build this option enables `go install` vs `go build`
which decreases incremental compilation time.
Note that this option is not supported in cross-compile builds.
For this reason the incremental build option must not be used for release builds.

[rkt-tpm]: devel/tpm.md
[trousers]: http://trousers.sourceforge.net/
