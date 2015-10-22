<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/devel/releasing.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Releasing Kubernetes

This document explains how to cut a release, and the theory behind it. If you
just want to cut a release and move on with your life, you can stop reading
after the first section.

## How to cut a Kubernetes release

Regardless of whether you are cutting a major or minor version, cutting a
release breaks down into four pieces:

1. selecting release components;
1. cutting/branching the release;
1. publishing binaries and release notes.

You should progress in this strict order.

### Selecting release components

First, figure out what kind of release you're doing, what branch you're cutting
from, and other prerequisites.

* Alpha releases (`vX.Y.0-alpha.W`) are cut directly from `master`.
  * Alpha releases don't require anything besides green tests, (see below).
* Official releases (`vX.Y.Z`) are cut from their respective release branch,
  `release-X.Y`.
  * Make sure all necessary cherry picks have been resolved.  You should ensure
    that all outstanding cherry picks have been reviewed and merged and the
    branch validated on Jenkins. See [Cherry Picks](cherry-picks.md) for more
    information on how to manage cherry picks prior to cutting the release.
  * Official releases also require green tests, (see below).
* New release series are also cut direclty from `master`.
  * **This is a big deal!**  If you're reading this doc for the first time, you
    probably shouldn't be doing this release, and should talk to someone on the
    release team.
  * New release series cut a new release branch, `release-X.Y`, off of
    `master`, and also release the first beta in the series, `vX.Y.0-beta`.
  * Every change in the `vX.Y` series from this point on will have to be
    cherry picked, so be sure you want to do this before proceeding.
  * You should still look for green tests, (see below).

No matter what you're cutting, you're going to want to look at
[Jenkins](http://go/k8s-test/).  Figure out what branch you're cutting from,
(see above,) and look at the critical jobs building from that branch.  First
glance through builds and look for nice solid rows of green builds, and then
check temporally with the other critical builds to make sure they're solid
around then as well. Once you find some greens, you can find the Git hash for a
build by looking at the Full Console Output and searching for `githash=`. You
should see a line:

```console
githash=v1.2.0-alpha.2.164+b44c7d79d6c9bb
```

Or, if you're cutting from a release branch (i.e. doing an official release),

```console
githash=v1.1.0-beta.567+d79d6c9bbb44c7
```

Because Jenkins builds frequently, if you're looking between jobs
(e.g. `kubernetes-e2e-gke-ci` and `kubernetes-e2e-gce`), there may be no single
`githash` that's been run on both jobs. In that case, take the a green
`kubernetes-e2e-gce` build (but please check that it corresponds to a temporally
similar build that's green on `kubernetes-e2e-gke-ci`). Lastly, if you're having
trouble understanding why the GKE continuous integration clusters are failing
and you're trying to cut a release, don't hesitate to contact the GKE
oncall.

Before proceeding to the next step:

```sh
export GITHASH=v1.2.0-alpha.2.164+b44c7d79d6c9bb
```

Where `v1.2.0-alpha.2.164+b44c7d79d6c9bb` is the Git hash you decided on. This
will become your release point.

### Cutting/branching the release

You'll need the latest version of the releasing tools:

```console
git clone git@github.com:kubernetes/contrib.git
cd contrib/release
```

#### Cutting an alpha release (`vX.Y.0-alpha.W`)

Figure out what version you're cutting, and

```console
export VER=vX.Y.0-alpha.W
```

then, from `contrib/release`, run

```console
cut-alpha.sh "${VER}" "${GITHASH}"
```

This will:

1. clone a temporary copy of the [kubernetes repo](https://github.com/kubernetes/kubernetes);
1. mark the `vX.Y.0-alpha.W` tag at the given Git hash;
1. push the tag to GitHub;
1. build the release binaries at the given Git hash;
1. publish the binaries to GCS;
1. prompt you to do the remainder of the work.

#### Cutting an official release (`vX.Y.Z`)

Figure out what version you're cutting, and

```console
export VER=vX.Y.Z
```

then, from `contrib/release`, run

```console
cut-official.sh "${VER}" "${GITHASH}"
```

This will:

1. clone a temporary copy of the [kubernetes repo](https://github.com/kubernetes/kubernetes);
1. do a series of commits on the branch, including forking the documentation
   and doing the release version commit;
  * TODO(ihmccreery) it's not yet clear what exactly this is going to look like.
1. mark both the `vX.Y.Z` and `vX.Y.(Z+1)-beta` tags at the given Git hash;
1. push the tags to GitHub;
1. build the release binaries at the given Git hash (on the appropriate
   branch);
1. publish the binaries to GCS;
1. prompt you to do the remainder of the work.

#### Branching a new release series (`vX.Y`)

Once again, **this is a big deal!**  If you're reading this doc for the first
time, you probably shouldn't be doing this release, and should talk to someone
on the release team.

Figure out what series you're cutting, and

```console
export VER=vX.Y
```

then, from `contrib/release`, run

```console
branch-series.sh "${VER}" "${GITHASH}"
```

This will:

1. clone a temporary copy of the [kubernetes repo](https://github.com/kubernetes/kubernetes);
1. mark the `vX.(Y+1).0-alpha.0` tag at the given Git hash on `master`;
1. fork a new branch `release-X.Y` off of `master` at the Given Git hash;
1. do a series of commits on the branch, including forking the documentation
   and doing the release version commit;
  * TODO(ihmccreery) it's not yet clear what exactly this is going to look like.
1. mark the `vX.Y.0-beta` tag at the appropriate commit on the new `release-X.Y` branch;
1. push the tags to GitHub;
1. build the release binaries at the appropriate Git hash on the appropriate
   branches, (for both the new alpha and beta releases);
1. publish the binaries to GCS;
1. prompt you to do the remainder of the work.

**TODO(ihmccreery)**: can we fix tags, etc., if you have to shift the release branchpoint?

### Publishing binaries and release notes

Whichever script you ran above will prompt you to take any remaining steps,
including publishing binaries and release notes.

**TODO(ihmccreery)**: deal with the `making-release-notes` doc in `docs/devel`.

## Origin of the Sources

TODO(ihmccreery) update this

Kubernetes may be built from either a git tree (using `hack/build-go.sh`) or
from a tarball (using either `hack/build-go.sh` or `go install`) or directly by
the Go native build system (using `go get`).

When building from git, we want to be able to insert specific information about
the build tree at build time. In particular, we want to use the output of `git
describe` to generate the version of Kubernetes and the status of the build
tree (add a `-dirty` prefix if the tree was modified.)

When building from a tarball or using the Go build system, we will not have
access to the information about the git tree, but we still want to be able to
tell whether this build corresponds to an exact release (e.g. v0.3) or is
between releases (e.g. at some point in development between v0.3 and v0.4).

## Version Number Format

TODO(ihmccreery) update everything below here

In order to account for these use cases, there are some specific formats that
may end up representing the Kubernetes version. Here are a few examples:

- **v0.5**: This is official version 0.5 and this version will only be used
  when building from a clean git tree at the v0.5 git tag, or from a tree
  extracted from the tarball corresponding to that specific release.
- **v0.5-15-g0123abcd4567**: This is the `git describe` output and it indicates
  that we are 15 commits past the v0.5 release and that the SHA1 of the commit
  where the binaries were built was `0123abcd4567`. It is only possible to have
  this level of detail in the version information when building from git, not
  when building from a tarball.
- **v0.5-15-g0123abcd4567-dirty** or **v0.5-dirty**: The extra `-dirty` prefix
  means that the tree had local modifications or untracked files at the time of
  the build, so there's no guarantee that the source code matches exactly the
  state of the tree at the `0123abcd4567` commit or at the `v0.5` git tag
  (resp.)
- **v0.5-dev**: This means we are building from a tarball or using `go get` or,
  if we have a git tree, we are using `go install` directly, so it is not
  possible to inject the git version into the build information. Additionally,
  this is not an official release, so the `-dev` prefix indicates that the
  version we are building is after `v0.5` but before `v0.6`. (There is actually
  an exception where a commit with `v0.5-dev` is not present on `v0.6`, see
  later for details.)

## Injecting Version into Binaries

In order to cover the different build cases, we start by providing information
that can be used when using only Go build tools or when we do not have the git
version information available.

To be able to provide a meaningful version in those cases, we set the contents
of variables in a Go source file that will be used when no overrides are
present.

We are using `pkg/version/base.go` as the source of versioning in absence of
information from git. Here is a sample of that file's contents:

```go
var (
    gitVersion   string = "v0.4-dev"  // version from git, output of $(git describe)
    gitCommit    string = ""          // sha1 from git, output of $(git rev-parse HEAD)
)
```

This means a build with `go install` or `go get` or a build from a tarball will
yield binaries that will identify themselves as `v0.4-dev` and will not be able
to provide you with a SHA1.

To add the extra versioning information when building from git, the
`hack/build-go.sh` script will gather that information (using `git describe` and
`git rev-parse`) and then create a `-ldflags` string to pass to `go install` and
tell the Go linker to override the contents of those variables at build time. It
can, for instance, tell it to override `gitVersion` and set it to
`v0.4-13-g4567bcdef6789-dirty` and set `gitCommit` to `4567bcdef6789...` which
is the complete SHA1 of the (dirty) tree used at build time.

## Release Notes

TODO(ihmccreery) update this

No official release should be made final without properly matching release notes.

There should be made available, per release, a small summary, preamble, of the
major changes, both in terms of feature improvements/bug fixes and notes about
functional feature changes (if any) regarding the previous released version so
that the BOM regarding updating to it gets as obvious and trouble free as possible.

After this summary, preamble, all the relevant PRs/issues that got in that
version should be listed and linked together with a small summary understandable
by plain mortals (in a perfect world PR/issue's title would be enough but often
it is just too cryptic/geeky/domain-specific that it isn't).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/releasing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
