# Releasing Kubernetes

This document explains how to create a Kubernetes release (as in version) and
how the version information gets embedded into the built binaries.

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

```
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

![Diagram of git commits involved in the release](./releasing.png)

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
this needs to be done from a git client and pushed to GitHub using SSH.)

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

```
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


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/releasing.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/devel/releasing.md?pixel)]()
