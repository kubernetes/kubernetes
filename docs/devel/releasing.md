<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Releasing Kubernetes

This document explains how to cut a release, and the theory behind it. If you
just want to cut a release and move on with your life, you can stop reading
after the first section.

## How to cut a Kubernetes release

Regardless of whether you are cutting a major or minor version, cutting a
release breaks down into four pieces:

1. Selecting release components.
1. Tagging and merging the release in Git.
1. Building and pushing the binaries.
1. Writing release notes.

You should progress in this strict order.

### Building a New Major/Minor Version (`vX.Y.0`)

#### Selecting Release Components

When cutting a major/minor release, your first job is to find the branch
point. We cut `vX.Y.0` releases directly from `master`, which is also the
branch that we have most continuous validation on. Go first to [the main GCE
Jenkins end-to-end job](http://go/k8s-test/job/kubernetes-e2e-gce) and next to [the
Critical Builds page](http://go/k8s-test/view/Critical%20Builds) and hopefully find a
recent Git hash that looks stable across at least `kubernetes-e2e-gce` and
`kubernetes-e2e-gke-ci`. First glance through builds and look for nice solid
rows of green builds, and then check temporally with the other Critical Builds
to make sure they're solid around then as well. Once you find some greens, you
can find the Git hash for a build by looking at the "Console Log", then look for
`githash=`. You should see a line line:

```console
+ githash=v0.20.2-322-g974377b
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
export BRANCHPOINT=v0.20.2-322-g974377b
```

Where `v0.20.2-322-g974377b` is the git hash you decided on. This will become
our (retroactive) branch point.

#### Branching, Tagging and Merging

Do the following:

1. `export VER=x.y` (e.g. `0.20` for v0.20)
1. cd to the base of the repo
1. `git fetch upstream && git checkout -b release-${VER} ${BRANCHPOINT}` (you did set `${BRANCHPOINT}`, right?)
1. Make sure you don't have any files you care about littering your repo (they
   better be checked in or outside the repo, or the next step will delete them).
1. `make clean && git reset --hard HEAD && git clean -xdf`
1. `make` (TBD: you really shouldn't have to do this, but the swagger output step requires it right now)
1. `./build/mark-new-version.sh v${VER}.0` to mark the new release and get further
   instructions. This creates a series of commits on the branch you're working
   on (`release-${VER}`), including forking our documentation for the release,
   the release version commit (which is then tagged), and the post-release
   version commit.
1. Follow the instructions given to you by that script. They are canon for the
   remainder of the Git process. If you don't understand something in that
   process, please ask!

**TODO**: how to fix tags, etc., if you have to shift the release branchpoint.

#### Building and Pushing Binaries

In your git repo (you still have `${VER}` set from above right?):

1. `git checkout upstream/master && build/build-official-release.sh v${VER}.0` (the `build-official-release.sh` script is version agnostic, so it's best to run it off `master` directly).
1. Follow the instructions given to you by that script.
1. At this point, you've done all the Git bits, you've got all the binary bits pushed, and you've got the template for the release started on GitHub.

#### Writing Release Notes

[This helpful guide](making-release-notes.md) describes how to write release
notes for a major/minor release. In the release template on GitHub, leave the
last PR number that the tool finds for the `.0` release, so the next releaser
doesn't have to hunt.

### Building a New Patch Release (`vX.Y.Z` for `Z > 0`)

#### Selecting Release Components

We cut `vX.Y.Z` releases from the `release-vX.Y` branch after all cherry picks
to the branch have been resolved. You should ensure all outstanding cherry picks
have been reviewed and merged and the branch validated on Jenkins (validation
TBD). See the [Cherry Picks](cherry-picks.md) for more information on how to
manage cherry picks prior to cutting the release.

#### Tagging and Merging

1. `export VER=x.y` (e.g. `0.20` for v0.20)
1. `export PATCH=Z` where `Z` is the patch level of `vX.Y.Z`
1. cd to the base of the repo
1. `git fetch upstream && git checkout -b upstream/release-${VER} release-${VER}`
1. Make sure you don't have any files you care about littering your repo (they
   better be checked in or outside the repo, or the next step will delete them).
1. `make clean && git reset --hard HEAD && git clean -xdf`
1. `make` (TBD: you really shouldn't have to do this, but the swagger output step requires it right now)
1. `./build/mark-new-version.sh v${VER}.${PATCH}` to mark the new release and get further
   instructions. This creates a series of commits on the branch you're working
   on (`release-${VER}`), including forking our documentation for the release,
   the release version commit (which is then tagged), and the post-release
   version commit.
1. Follow the instructions given to you by that script. They are canon for the
   remainder of the Git process. If you don't understand something in that
   process, please ask! When proposing PRs, you can pre-fill the body with
   `hack/cherry_pick_list.sh upstream/release-${VER}` to inform people of what
   is already on the branch.

**TODO**: how to fix tags, etc., if the release is changed.

#### Building and Pushing Binaries

In your git repo (you still have `${VER}` and `${PATCH}` set from above right?):

1. `git checkout upstream/master && build/build-official-release.sh
   v${VER}.${PATCH}` (the `build-official-release.sh` script is version
   agnostic, so it's best to run it off `master` directly).
1. Follow the instructions given to you by that script. At this point, you've
   done all the Git bits, you've got all the binary bits pushed, and you've got
   the template for the release started on GitHub.

#### Writing Release Notes

Run `hack/cherry_pick_list.sh ${VER}.${PATCH}~1` to get the release notes for
the patch release you just created. Feel free to prune anything internal, like
you would for a major release, but typically for patch releases we tend to
include everything in the release notes.

## Origin of the Sources

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

## Handling Official Versions

Handling official versions from git is easy, as long as there is an annotated
git tag pointing to a specific version then `git describe` will return that tag
exactly which will match the idea of an official version (e.g. `v0.5`).

Handling it on tarballs is a bit harder since the exact version string must be
present in `pkg/version/base.go` for it to get embedded into the binaries. But
simply creating a commit with `v0.5` on its own would mean that the commits
coming after it would also get the `v0.5` version when built from tarball or `go
get` while in fact they do not match `v0.5` (the one that was tagged) exactly.

To handle that case, creating a new release should involve creating two adjacent
commits where the first of them will set the version to `v0.5` and the second
will set it to `v0.5-dev`. In that case, even in the presence of merges, there
will be a single commit where the exact `v0.5` version will be used and all
others around it will either have `v0.4-dev` or `v0.5-dev`.

The diagram below illustrates it.

![Diagram of git commits involved in the release](releasing.png)

After working on `v0.4-dev` and merging PR 99 we decide it is time to release
`v0.5`. So we start a new branch, create one commit to update
`pkg/version/base.go` to include `gitVersion = "v0.5"` and `git commit` it.

We test it and make sure everything is working as expected.

Before sending a PR for it, we create a second commit on that same branch,
updating `pkg/version/base.go` to include `gitVersion = "v0.5-dev"`. That will
ensure that further builds (from tarball or `go install`) on that tree will
always include the `-dev` prefix and will not have a `v0.5` version (since they
do not match the official `v0.5` exactly.)

We then send PR 100 with both commits in it.

Once the PR is accepted, we can use `git tag -a` to create an annotated tag
*pointing to the one commit* that has `v0.5` in `pkg/version/base.go` and push
it to GitHub. (Unfortunately GitHub tags/releases are not annotated tags, so
this needs to be done from a git client and pushed to GitHub using SSH or
HTTPS.)

## Parallel Commits

While we are working on releasing `v0.5`, other development takes place and
other PRs get merged. For instance, in the example above, PRs 101 and 102 get
merged to the master branch before the versioning PR gets merged.

This is not a problem, it is only slightly inaccurate that checking out the tree
at commit `012abc` or commit `345cde` or at the commit of the merges of PR 101
or 102 will yield a version of `v0.4-dev` *but* those commits are not present in
`v0.5`.

In that sense, there is a small window in which commits will get a
`v0.4-dev` or `v0.4-N-gXXX` label and while they're indeed later than `v0.4`
but they are not really before `v0.5` in that `v0.5` does not contain those
commits.

Unfortunately, there is not much we can do about it. On the other hand, other
projects seem to live with that and it does not really become a large problem.

As an example, Docker commit a327d9b91edf has a `v1.1.1-N-gXXX` label but it is
not present in Docker `v1.2.0`:

```console
$ git describe a327d9b91edf
v1.1.1-822-ga327d9b91edf

$ git log --oneline v1.2.0..a327d9b91edf
a327d9b91edf Fix data space reporting from Kb/Mb to KB/MB

(Non-empty output here means the commit is not present on v1.2.0.)
```

## Release Notes

No official release should be made final without properly matching release notes.

There should be made available, per release, a small summary, preamble, of the
major changes, both in terms of feature improvements/bug fixes and notes about
functional feature changes (if any) regarding the previous released version so
that the BOM regarding updating to it gets as obvious and trouble free as possible.

After this summary, preamble, all the relevant PRs/issues that got in that
version should be listed and linked together with a small summary understandable
by plain mortals (in a perfect world PR/issue's title would be enough but often
it is just too cryptic/geeky/domain-specific that it isn't).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/releasing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
