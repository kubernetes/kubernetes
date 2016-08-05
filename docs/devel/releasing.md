<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/devel/releasing.md).

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
1. updating the master branch.

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
[Jenkins](http://kubekins.dls.corp.google.com/) (Google internal only).  Figure
out what branch you're cutting from, (see above,) and look at the critical jobs
building from that branch.  First glance through builds and look for nice solid
rows of green builds, and then check temporally with the other critical builds
to make sure they're solid around then as well.

If you're doing an alpha release or cutting a new release series, you can
choose an arbitrary build.  If you are doing an official release, you have to
release from HEAD of the branch, (because you have to do some version-rev
commits,) so choose the latest build on the release branch.  (Remember, that
branch should be frozen.)

Once you find some greens, you can find the build hash for a build by looking at
the Full Console Output and searching for `build_version=`. You should see a line:

```console
build_version=v1.2.0-alpha.2.164+b44c7d79d6c9bb
```

Or, if you're cutting from a release branch (i.e. doing an official release),

```console
build_version=v1.1.0-beta.567+d79d6c9bbb44c7
```

Please note that `build_version` was called `githash` versions prior to v1.2.

Because Jenkins builds frequently, if you're looking between jobs
(e.g. `kubernetes-e2e-gke-ci` and `kubernetes-e2e-gce`), there may be no single
`build_version` that's been run on both jobs. In that case, take the a green
`kubernetes-e2e-gce` build (but please check that it corresponds to a temporally
similar build that's green on `kubernetes-e2e-gke-ci`). Lastly, if you're having
trouble understanding why the GKE continuous integration clusters are failing
and you're trying to cut a release, don't hesitate to contact the GKE
oncall.

Before proceeding to the next step:

```sh
export BUILD_VERSION=v1.2.0-alpha.2.164+b44c7d79d6c9bb
```

Where `v1.2.0-alpha.2.164+b44c7d79d6c9bb` is the build hash you decided on. This
will become your release point.

### Cutting/branching the release

You'll need the latest version of the releasing tools:

```console
git clone git@github.com:kubernetes/kubernetes.git
cd kubernetes
```

or `git fetch upstream && git checkout upstream/master` from an existing repo.

Decide what version you're cutting and export it:

- alpha release: `export RELEASE_VERSION="vX.Y.0-alpha.W"`;
- beta release: `export RELEASE_VERSION="vX.Y.Z-beta.W"`;
- official release: `export RELEASE_VERSION="vX.Y.Z"`;
- new release series: `export RELEASE_VERSION="vX.Y"`.

Then, run

```console
./release/cut-official-release.sh "${RELEASE_VERSION}" "${BUILD_VERSION}"
```

This will do a dry run of the release.  It will give you instructions at the
end for `pushd`ing into the dry-run directory and having a look around.
`pushd` into the directory and make sure everything looks as you expect:

```console
git log "${RELEASE_VERSION}"  # do you see the commit you expect?
make release
./cluster/kubectl.sh version -c
```

If you're satisfied with the result of the script, go back to `upstream/master`
run

```console
./release/cut-official-release.sh "${RELEASE_VERSION}" "${BUILD_VERSION}" --no-dry-run
```

and follow the instructions.

### Publishing binaries and release notes

Only publish a beta release if it's a standalone pre-release (*not*
vX.Y.Z-beta.0).  We create beta tags after we do official releases to
maintain proper semantic versioning, but we don't publish these beta releases.

The script you ran above will prompt you to take any remaining steps to push
tars, and will also give you a template for the release notes.  Compose an
email to the team with the template.  Figure out what the PR numbers for this
release and last release are, and get an api-token from GitHub
(https://github.com/settings/tokens).  From a clone of
[kubernetes/contrib](https://github.com/kubernetes/contrib),

```
go run release-notes/release-notes.go --last-release-pr=<number> --current-release-pr=<number> --api-token=<token> --base=<release-branch>
```

where `<release-branch>` is `master` for alpha releases and `release-X.Y` for beta and official releases.

**If this is a first official release (vX.Y.0)**, look through the release
notes for all of the alpha releases since the last cycle, and include anything
important in release notes.

Feel free to edit the notes, (e.g. cherry picks should generally just have the
same title as the original PR).

Send the email out, letting people know these are the draft release notes.  If
they want to change anything, they should update the appropriate PRs with the
`release-note` label.

When you're ready to announce the release, [create a GitHub
release](https://github.com/kubernetes/kubernetes/releases/new):

1. pick the appropriate tag;
1. check "This is a pre-release" if it's an alpha or beta release;
1. fill in the release title from the draft;
1. re-run the appropriate release notes tool(s) to pick up any changes people
   have made;
1. find the appropriate `kubernetes.tar.gz` in [GCS bucket](https://console.developers.google.com/storage/browser/kubernetes-release/release/),
   download it, double check the hash (compare to what you had in the release
   notes draft), and attach it to the release; and
1. publish!

### Manual tasks for new release series

*TODO(#20946) Burn this list down.*

If you are cutting a new release series, there are a few tasks that haven't yet
been automated that need to happen after the branch has been cut:

1. Update the master branch constant for doc generation: change the
   `latestReleaseBranch` in `cmd/mungedocs/mungedocs.go` to the new release
   branch (`release-X.Y`), run `hack/update-generated-docs.sh`.  This will let
   the unversioned warning in docs point to the latest release series. Please
   send the changes as a PR titled "Update the latestReleaseBranch to
   release-X.Y in the munger".
1. Send a note to the test team (@kubernetes/goog-testing) that a new branch
   has been created.
   1. There is currently much work being done on our Jenkins infrastructure
      and configs.  Eventually we could have a relatively simple interface
      to make this change or a way to automatically use the new branch.
      See [recent Issue #22672](https://github.com/kubernetes/kubernetes/issues/22672).
   1. You can provide this guidance in the email to aid in the setup:
      1. See [End-2-End Testing in Kubernetes](e2e-tests.md) for the test jobs
         that should be running in CI, which are under version control in
         `hack/jenkins/e2e.sh` (on the release branch) and
         `hack/jenkins/job-configs/kubernetes-jenkins/kubernetes-e2e.yaml`
         (in `master`).  You'll want to munge these for the release
         branch so that, as we cherry-pick fixes onto the branch, we know that
         it builds, etc.  (Talk with @ihmccreery for more details.)
      1. Make sure all features that are supposed to be GA are covered by tests,
         but remove feature tests on the release branch for features that aren't
         GA.  You can use `hack/list-feature-tests.sh` to see a list of tests
         labeled as `[Feature:.+]`; make sure that these are all either
         covered in CI jobs on the release branch or are experimental
         features.  (The answer should already be 'yes', but this is a
         good time to reconcile.)
      1. Make a dashboard in Jenkins that contains all of the jobs for this
         release cycle, and also add them to Critical Builds.  (Don't add
         them to the merge-bot blockers; see kubernetes/contrib#156.)


## Injecting Version into Binaries

*Please note that this information may be out of date.  The scripts are the
authoritative source on how version injection works.*

Kubernetes may be built from either a git tree or from a tarball.  We use
`make` to encapsulate a number of build steps into a single command.  This
includes generating code, which means that tools like `go build` might work
(once files are generated) but might be using stale generated code.  `make` is
the supported way to build.

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
`make` build will gather that information (using `git describe` and
`git rev-parse`) and then create a `-ldflags` string to pass to `go install` and
tell the Go linker to override the contents of those variables at build time. It
can, for instance, tell it to override `gitVersion` and set it to
`v0.4-13-g4567bcdef6789-dirty` and set `gitCommit` to `4567bcdef6789...` which
is the complete SHA1 of the (dirty) tree used at build time.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/releasing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
