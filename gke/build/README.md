# GKE Build Entrypoint

Design doc: go/gke-k8s-build-isolation

TL;DR: The `make_*.sh` scripts here are maintained exclusively by Google to
ensure a consistent, smooth experience for building Kubernetes artifacts.

# Overview

Virtually all of the heavy lifting is inside [`lib_gke.sh`](./lib_gke.sh). This
library file has functions that consume the build configuration YAMLs which are
defined in `./config/*.yaml`. These YAMLs determine all of the knobs that can be
tweaked to modify the build.

All of the files named `make_*.sh` are the consumers of this library, and begin
execution by calling into `gke_build_entrypoint()`. This function provides some
additional settings that can be provided at the command line, such as
`SKIP_DOCKER` to skip generation of Docker images. For a full list of such
options, see the `process_args()` function.

`lib_gke.sh` calls into OSS code to ultimately perform some of the actions.
Eventually however, the goal is to reduce this "OSS surface" as much as possible
to guarantee that breakages in OSS do not spill over to us.

## Louhi

[Louhi](louhi) is one of the main consumers of this build system. See
[`make_louhi_prod.sh`](./make_louhi_prod.sh) to see how Louhi builds are
produced. The official GKE Louhi Flow that calls into `make_louhi_wrapper.sh` is
[here][louhi-flow-official].

Users must modify `lib_louhi.sh` instead of modifying the [Louhi Stage
Type](louhi-stagetype-official) for any new changes to the Louhi builds.

# Usage

The following is a list of consumers and the recommended guidance for using this
build system:

- **Louhi**: Use either `./make_louhi_prod.sh` or `./make_louhi_test.sh`.
- **Developers**: There are two options.
  1. Use `./make_dev_simple.sh` to get a quick build. This build skips building
    Docker images. If you want to still build Docker images, invoke with
    `./make_dev_simple.sh SKIP_DOCKER=0`.
  2. If you want to use your build to create a new cluster, run
    `./make_dev_push.sh` instead. This is like `make_dev_simple.sh`, but
    additionally runs the `package,validate,push-gcs` steps to generate all
    packages including tarballs and Docker images (the images are saved into
    tarballs as well), and push them to GCS.
- **CI/Prow**: Use `./make_ci.sh`.
- **Others**: Use `./make_custom.sh`. This provides "raw" access to all settings
  that can be tweaked. However, if you find yourself using a particular set of
  settings repeatedly, you are encouraged to add a new `./make_*` entrypoint to
  capture these frequently-used settings.

## Arguments/Options

All of the `./make_*` scripts in this directory can take certain arguments
(defined in `process_args()` in `lib_gke.sh`). To simplify the design, there is
no distinction between arguments and options, and they all take the form
`ARGUMENT=FOO` where `ARGUMENT` is the knob you want to modify for the build.
For example, to skip generation of Docker images during the `package` step, you
can do `./make_custom.sh GKE_BUILD_ACTIONS=compile,package SKIP_DOCKER=1`.

Below is a list of supported options and their descriptions.

- `GKE_BUILD_CONFIG`: CSV of paths to YAMLs to read for configuration. Later
  paths override earlier ones. Thus, you can use
  `GKE_BUILD_CONFIG=gke/build/config/common.yaml,gke/build/config/foo.yaml` to
  tweak `common.yaml` with `foo.yaml`.
- `GKE_BUILD_ACTIONS`: CSV of build actions. Supported actions are `compile`,
  `package`, `validate`, `push-gcs`, `push-gcr`, and `print-version`. If you
  only want to compile, package, and validate, for example, you can do
  `GKE_BUILD_ACTIONS=compile,package,validate`. Below are the actions and what
  they mean:
  - `compile`: Compile all Go binaries.
  - `package`: Creates tarballs (build artifacts) and also Docker images in the
    local Docker daemon.
  - `validate`: Validates tarballs to look for things like licenses and source
    code for compliance.
  - `print-version`: Prints the version to be used for the build to STDOUT and
    exits (does nothing else). E.g., `./make_dev_simple.sh
    GKE_BUILD_ACTIONS=print-version` with an optional `VERSION_SUFFIX=...`. By
    default there is a lot of logging to STDERR, so if you want to visually mask
    that out, do `2>/dev/null ./make_dev_simple.sh ...`.
  - `enter-build-container`: Runs `docker run -it ...` to enter an interactive
    Bash shell inside the build image specified in the `GKE_BUILD_CONFIG`.
    Implied with `ENTER_BUILD_CONTAINER=1`.
- `VERSION`: The version string to use. If this is not set, `git` is used to
  auto-generate the version. To set it to a different string, use `VERSION=v...`
  (notice the leading `v`). During the `compile` step, this string gets embedded
  into binaries directly (e.g., `kubectl version` will report it). During the
  `validate` and `push-*` steps, this argument can be used to select which
  version to validate or push (assuming that there have already been multiple
  `compile,package` runs). To see the compiled and packaged versions (builds),
  go to `<KUBE_ROOT>/_output/for-gcs/`, where each symlinked folder here will
  contain all build artifacts.
- `VERSION_SUFFIX`: Version suffix to append. This is optional, and will be
  empty by default. If given, it will be appended to the end of the version
  string with a dash, as in `<VERSION>-<VERSION_SUFFIX>`. A typical use case is
  `VERSION_SUFFIX=${USER}` to append your username to the end of the version
  string artifact.
- `INJECT_DEV_VERSION_MARKER`: Decides whether to inject a '-gke.99.99+' string.
  Used to guarantee that the final build artifacts look like CI/dev builds.
- `TARGET_PLATFORMS`: CSV of different platforms to for all build actions. E.g.,
  `make_custom.sh TARGET_PLATFORMS=linux/amd64,linux/arm64,windows/amd64`.
- `GCR_REPO`: The GCR repository to name images, during the `package` step. If
  defined, it will override anything set in the `package.gcr.repo` field in the
  CSV of YAMLs defined in `GKE_BUILD_CONFIG`.
- `GCS_BUCKET`: The GCS bucket to push up the GCS artifacts into. To be precise,
  everything under `<KUBE_ROOT>/_output/for-gcs/<VERSION>` will be uploaded into
  this bucket. If blank, the setting in `push.gcs.to` in the config YAML will
  take precedence.
- `SKIP_DOCKER`: Whether to skip generating Docker images during the `package`
  step. Can be `1` for true and `0` for false. Default false.
- `TEMP_DIR`: The temporary directory to use for storing build artifacts. By
  default, all build artifacts are stored in `/tmp`. This useful to set if your
  `/` partition is nearly full.
- `ENTER_BUILD_CONTAINER`: Enter an interactive Bash shell inside the build
  image specified in `GKE_BUILD_CONFIG`. Can be `1` for true and `0` for false.
  Default false. If true, implies `GKE_BUILD_ACTIONS=enter-build-container`.

# Extending

New consumers of this system should:

- create a `lib_<consumer>.sh` library (use [`lib_louhi.sh`](./lib_louhi.sh) as an example), and
- create a `make_<consumer>.sh` entrypoint for their particular use case.

[louhi]: http://go/louhi
[louhi-flow-official]: https://louhi.dev/?projectId=5846944631226368&expandedFlows=ff455683-a314-4346-9211-f6928045ecae#/flows
[louhi-stagetype-official]: https://louhi.dev/?projectId=5846944631226368#/stage-type/f30a4ed0-ef3d-4c55-be71-5cb28d88373e
