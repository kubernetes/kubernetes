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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.2/docs/devel/development.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Development Guide

This document is intended to be the canonical source of truth for things like
supported toolchain versions for building Kubernetes. If you find a
requirement that this doc does not capture, please file a bug. If you find
other docs with references to requirements that are not simply links to this
doc, please file a bug.

This document is intended to be relative to the branch in which it is found.
It is guaranteed that requirements will change over time for the development
branch, but release branches of Kubernetes should not change.

## Building Kubernetes

Official releases are built using Docker containers. To build Kubernetes using
Docker please follow [these instructions](http://releases.k8s.io/HEAD/build/README.md).

### Go development environment

Kubernetes is written in the [Go](http://golang.org) programming language.
To build Kubernetes without using Docker containers, you'll need a Go
development environment. Builds for Kubernetes 1.0 - 1.2 require Go version
1.4.2. Builds for Kubernetes 1.3 and higher require Go version 1.6.0. If you
haven't set up a Go development environment, please follow [these instructions](http://golang.org/doc/code.html)
to install the go tools and set up a GOPATH.

To build Kubernetes using your local Go development environment (generate linux
binaries):

        hack/build-go.sh
You may pass build options and packages to the script as necessary. To build binaries for all platforms:

        hack/build-cross.sh

## Workflow

Below, we outline one of the more common git workflows that core developers use.
Other git workflows are also valid.

### Visual overview

![Git workflow](git_workflow.png)

### Fork the main repository

1. Go to https://github.com/kubernetes/kubernetes
2. Click the "Fork" button (at the top right)

### Clone your fork

The commands below require that you have $GOPATH set ([$GOPATH docs](https://golang.org/doc/code.html#GOPATH)). We highly recommend you put Kubernetes' code into your GOPATH. Note: the commands below will not work if
there is more than one directory in your `$GOPATH`.

```sh
mkdir -p $GOPATH/src/k8s.io
cd $GOPATH/src/k8s.io
# Replace "$YOUR_GITHUB_USERNAME" below with your github username
git clone https://github.com/$YOUR_GITHUB_USERNAME/kubernetes.git
cd kubernetes
git remote add upstream 'https://github.com/kubernetes/kubernetes.git'
```

### Create a branch and make changes

```sh
git checkout -b myfeature
# Make your code changes
```

### Keeping your development fork in sync

```sh
git fetch upstream
git rebase upstream/master
```

Note: If you have write access to the main repository at github.com/kubernetes/kubernetes, you should modify your git configuration so that you can't accidentally push to upstream:

```sh
git remote set-url --push upstream no_push
```

### Committing changes to your fork

Before committing any changes, please link/copy these pre-commit hooks into your .git
directory. This will keep you from accidentally committing non-gofmt'd Go code.

```sh
cd kubernetes/.git/hooks/
ln -s ../../hooks/pre-commit .
```

Then you can commit your changes and push them to your fork:

```sh
git commit
git push -f origin myfeature
```

### Creating a pull request

1. Visit https://github.com/$YOUR_GITHUB_USERNAME/kubernetes
2. Click the "Compare and pull request" button next to your "myfeature" branch.
3. Check out the pull request [process](pull-requests.md) for more details

### When to retain commits and when to squash

Upon merge, all git commits should represent meaningful milestones or units of
work.  Use commits to add clarity to the development and review process.

Before merging a PR, squash any "fix review feedback", "typo", and "rebased"
sorts of commits. It is not imperative that every commit in a PR compile and
pass tests independently, but it is worth striving for. For mass automated
fixups (e.g. automated doc formatting), use one or more commits for the
changes to tooling and a final commit to apply the fixup en masse. This makes
reviews much easier.

See [Faster Reviews](faster_reviews.md) for more details.

## Dependency management

Kubernetes uses [glide](https://github.com/Masterminds/glide) to manage
dependencies. It is not required for building Kubernetes but it is required
when managing dependencies under the `vendor` directory, and is required by a
number of the build and test scripts. Please make sure that `glide` is
installed and in your `$PATH`.

### Installing glide

There are many ways to build and host go binaries. Here is an easy way to get utilities like `glide` installed:

1) Decide on a `GOPATH` for tools.  You can use your existing `GOPATH` or create a new one.

```sh
export GOPATH=$HOME/go-tools
mkdir -p $GOPATH
go get github.com/Masterminds/glide
```

2) Add `$GOPATH/bin` to your path. Typically you'd add this to your `~/.profile`:

```sh
export GOPATH=$HOME/go-tools
export PATH=$PATH:$GOPATH/bin
```

### How vendored dependencies work in Kubernetes

Depending on which version of Go you are using, dependencies will be used in
one of a number of ways.  The vendored code is stored in the top-level `vendor`
directory, which is the "new" way to do vendored dependencies in Go, but you
can still use older Go versions.

If you use go-1.4 or go-1.5 with `GO15VENDOREXPERIMENT` unset, you have two
choices.  You can load all of the project's dependencies into your `GOPATH` or
you can use our build scripts which will synthesize an extra `GOPATH` entry for
you.

Option 1) You can copy deps into your `GOPATH` using `glide`.  From within the
Kubernetes source directory, run:

```sh
glide update --update-vendored --strip-vendor --strip-vcs --cache-gopath
```

This will consult our manifest of vendored dependencies, fetch them all from
version-control, and save them in your GOPATH.  Unfortunately, the way `glide`
works is that this will try to update your local repo.  There should be no
changes (or something is very wrong), but the `glide.lock` file may be updated
with a new timestamp.  it is safe to revert this change `git co -- glide.lock`.
We need the `--strip-vendor` flag so that projects which have their own
vendored deps will instead use the revisions we have chosen (for more
consistent builds).  We ned the `--strip-vcs` flag because `glide` demands it
when using `--strip-vendor`.

Option 2) You can use our build tools.  If you simply run `make` or
`hack/build-go.sh` or `make test` or `hack/test-go.sh` or (any of the related
make rules or build scripts) we will create a temporary dir and symlink it to
look like a `GOPATH` dir, add that to your `GOPATH` before running any `go`
commands.  This way is simpler, but somewhat slower.

If you use go-1.6 (or late)r or go-1.5 with `GO15VENDOREXPERIMENT=1` set, the
`go` commands should find the vendored code automatically.

### Using glide to manage dependencies

Here's a quick walkthrough of how to use glide to add or update a Kubernetes dependency.

```sh
glide get --update-vendored --strip-vendor --strip-vcs example.com/new-dep
```

This will pull the `example.com/new-dep` repo into the `vendor` directory along
with any dependencies it has (vendored or just imported) and then analyze the
codebase for any changes in the dependency graph.  It can take a while.

At the end of this you should have the new dependency and its recursive
dependencies in your `vendor` directory, as well as updated `glide.yaml` and
glide.lock` files.

Before sending a PR, it's a good idea to sanity check that your change is ok by
running `hack/verify-vendored-deps.sh`

Please send dependency updates in separate commits within your PR, for easier
reviewing.

FIXME: need to handle licenses with glide.
6) If you updated the Godeps, please also update `Godeps/LICENSES` by running `hack/update-godep-licenses.sh`.
FIXME: need to doc removing a dep

## Testing

Three basic commands let you run unit, integration and/or e2e tests:

```sh
cd kubernetes
hack/test-go.sh  # Run unit tests
hack/test-integration.sh  # Run integration tests, requires etcd
go run hack/e2e.go -v --build --up --test --down  # Run e2e tests
```

See the [testing guide](testing.md) for additional information and scenarios.

## Regenerating the CLI documentation

```sh
hack/update-generated-docs.sh
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/development.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
