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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/devel/releasing.md).

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
1. building and pushing the binaries; and
1. publishing binaries and release notes.

You should progress in this strict order.

### Selecting release components

First, figure out what kind of release you're doing, what branch you're cutting
from, and other prerequisites.

* Alpha releases (`vX.Y.0-alpha.W`) are cut directly from `master`.
  * Alpha releases don't require anything besides green tests, (see below).
* Beta releases (`vX.Y.Z-beta.W`) are cut from their respective release branch,
  `release-X.Y`.
  * Make sure all necessary cherry picks have been resolved.  You should ensure
    that all outstanding cherry picks have been reviewed and merged and the
    branch validated on Jenkins. See [Cherry Picks](cherry-picks.md) for more
    information on how to manage cherry picks prior to cutting the release.
  * Beta releases also require green tests, (see below).
* Official releases (`vX.Y.Z`) are cut from their respective release branch,
  `release-X.Y`.
  * Official releases should be similar or identical to their respective beta
    releases, so have a look at the cherry picks that have been merged since
    the beta release and question everything you find.
  * Official releases also require green tests, (see below).
* New release series are also cut directly from `master`.
  * **This is a big deal!**  If you're reading this doc for the first time, you
    probably shouldn't be doing this release, and should talk to someone on the
    release team.
  * New release series cut a new release branch, `release-X.Y`, off of
    `master`, and also release the first beta in the series, `vX.Y.0-beta.0`.
  * Every change in the `vX.Y` series from this point on will have to be
    cherry picked, so be sure you want to do this before proceeding.
  * You should still look for green tests, (see below).

No matter what you're cutting, you're going to want to look at
[Jenkins](http://go/k8s-test/).  Figure out what branch you're cutting from,
(see above,) and look at the critical jobs building from that branch.  First
glance through builds and look for nice solid rows of green builds, and then
check temporally with the other critical builds to make sure they're solid
around then as well.

If you're doing an alpha release or cutting a new release series, you can
choose an arbitrary build.  If you are doing an official release, you have to
release from HEAD of the branch, (because you have to do some version-rev
commits,) so choose the latest build on the release branch.  (Remember, that
branch should be frozen.)

Once you find some greens, you can find the git hash for a build by looking at
the Full Console Output and searching for `githash=`. You should see a line:

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

Where `v1.2.0-alpha.2.164+b44c7d79d6c9bb` is the git hash you decided on. This
will become your release point.

### Cutting/branching the release

You'll need the latest version of the releasing tools:

```console
git clone git@github.com:kubernetes/kubernetes.git
cd kubernetes
```

or `git checkout upstream/master` from an existing repo.

#### Cutting an alpha release (`vX.Y.0-alpha.W`)

Figure out what version you're cutting, and

```console
export VER="vX.Y.0-alpha.W"
```

then, run

```console
./release/cut-official-release.sh "${VER}" "${GITHASH}"
```

This will do a dry run of:

1. mark the `vX.Y.0-alpha.W` tag at the given git hash;
1. prompt you to do the remainder of the work, including building the
   appropriate binaries and pushing them to the appropriate places.

If you're satisfied with the result, run

```console
./release/cut-official-release.sh "${VER}" "${GITHASH}" --no-dry-run
```

and follow the instructions.

#### Cutting an beta release (`vX.Y.Z-beta.W`)

Figure out what version you're cutting, and

```console
export VER="vX.Y.Z-beta.W"
```

then, run

```console
./release/cut-official-release.sh "${VER}" "${GITHASH}"
```

This will do a dry run of:

1. do a series of commits on the release branch for `vX.Y.Z-beta.W`;
1. mark the `vX.Y.Z-beta.W` tag at the beta version commit;
1. prompt you to do the remainder of the work, including building the
   appropriate binaries and pushing them to the appropriate places.

If you're satisfied with the result, run

```console
./release/cut-official-release.sh "${VER}" "${GITHASH}" --no-dry-run
```

and follow the instructions.

#### Cutting an official release (`vX.Y.Z`)

Figure out what version you're cutting, and

```console
export VER="vX.Y.Z"
```

then, run

```console
./release/cut-official-release.sh "${VER}" "${GITHASH}"
```

This will do a dry run of:

1. do a series of commits on the branch for `vX.Y.Z`;
1. mark the `vX.Y.Z` tag at the release version commit;
1. do a series of commits on the branch for `vX.Y.(Z+1)-beta.0` on top of the
   previous commits;
1. mark the `vX.Y.(Z+1)-beta.0` tag at the beta version commit;
1. prompt you to do the remainder of the work, including building the
   appropriate binaries and pushing them to the appropriate places.

If you're satisfied with the result, run

```console
./release/cut-official-release.sh "${VER}" "${GITHASH}" --no-dry-run
```

and follow the instructions.

#### Branching a new release series (`vX.Y`)

Once again, **this is a big deal!**  If you're reading this doc for the first
time, you probably shouldn't be doing this release, and should talk to someone
on the release team.

Figure out what series you're cutting, and

```console
export VER="vX.Y"
```

then, run

```console
./release/cut-official-release.sh "${VER}" "${GITHASH}"
```

This will do a dry run of:

1. mark the `vX.(Y+1).0-alpha.0` tag at the given git hash on `master`;
1. fork a new branch `release-X.Y` off of `master` at the given git hash;
1. do a series of commits on the branch for `vX.Y.0-beta.0`;
1. mark the `vX.Y.0-beta.0` tag at the beta version commit;
1. prompt you to do the remainder of the work, including building the
   appropriate binaries and pushing them to the appropriate places.

If you're satisfied with the result, run

```console
./release/cut-official-release.sh "${VER}" "${GITHASH}" --no-dry-run
```

and follow the instructions.

### Publishing binaries and release notes

The script you ran above will prompt you to take any remaining steps, including
publishing binaries and release notes.

## Injecting Version into Binaries

*Please note that this information may be out of date.  The scripts are the
authoritative source on how version injection works.*

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


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/releasing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
