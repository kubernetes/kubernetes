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
Docker please follow [these
instructions](http://releases.k8s.io/HEAD/build/README.md).

### Go development environment

Kubernetes is written in the [Go](http://golang.org) programming language.
To build Kubernetes without using Docker containers, you'll need a Go
development environment. Builds for Kubernetes 1.0 - 1.2 require Go version
1.4.2. Builds for Kubernetes 1.3 and higher require Go version 1.6.0. If you
haven't set up a Go development environment, please follow [these
instructions](http://golang.org/doc/code.html) to install the go tools and set
up a GOPATH.

To build Kubernetes using your local Go development environment (generate linux
binaries):

        hack/build-go.sh
You may pass build options and packages to the script as necessary. To build
binaries for all platforms:

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

The commands below require that you have $GOPATH set ([$GOPATH
docs](https://golang.org/doc/code.html#GOPATH)). We highly recommend you put
Kubernetes' code into your GOPATH. Note: the commands below will not work if
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

Note: If you have write access to the main repository at
github.com/kubernetes/kubernetes, you should modify your git configuration so
that you can't accidentally push to upstream:

```sh
git remote set-url --push upstream no_push
```

### Committing changes to your fork

Before committing any changes, please link/copy the pre-commit hook into your
.git directory. This will keep you from accidentally committing non-gofmt'd Go
code. This hook will also do a build and test whether documentation generation
scripts need to be executed.

The hook requires both Godep and etcd on your `PATH`.

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

## godep and dependency management

Kubernetes uses [godep](https://github.com/tools/godep) to manage dependencies.
It is not strictly required for building Kubernetes but it is required when
managing dependencies under the Godeps/ tree, and is required by a number of the
build and test scripts. Please make sure that ``godep`` is installed and in your
``$PATH``.

### Installing godep

There are many ways to build and host Go binaries. Here is an easy way to get
utilities like `godep` installed:

1) Ensure that [mercurial](http://mercurial.selenic.com/wiki/Download) is
installed on your system. (some of godep's dependencies use the mercurial
source control system). Use `apt-get install mercurial` or `yum install
mercurial` on Linux, or [brew.sh](http://brew.sh) on OS X, or download directly
from mercurial.

2) Create a new GOPATH for your tools and install godep:

```sh
export GOPATH=$HOME/go-tools
mkdir -p $GOPATH
go get github.com/tools/godep
```

3) Add $GOPATH/bin to your path. Typically you'd add this to your ~/.profile:

```sh
export GOPATH=$HOME/go-tools
export PATH=$PATH:$GOPATH/bin
```

Note:
At this time, godep update in the Kubernetes project only works properly if your
version of godep is < 54.

To check your version of godep:

```sh
$ godep version
godep v53 (linux/amd64/go1.5.3)
```

### Using godep

Here's a quick walkthrough of one way to use godeps to add or update a
Kubernetes dependency into `vendor/`. For more details, please see the
instructions in [godep's documentation](https://github.com/tools/godep).

1) Devote a directory to this endeavor:

_Devoting a separate directory is not required, but it is helpful to separate
dependency updates from other changes._

```sh
export KPATH=$HOME/code/kubernetes
mkdir -p $KPATH/src/k8s.io/kubernetes
cd $KPATH/src/k8s.io/kubernetes
git clone https://path/to/your/fork .
# Or copy your existing local repo here. IMPORTANT: making a symlink doesn't work.
```

2) Set up your GOPATH.

```sh
# Option A: this will let your builds see packages that exist elsewhere on your system.
export GOPATH=$KPATH:$GOPATH
# Option B: This will *not* let your local builds see packages that exist elsewhere on your system.
export GOPATH=$KPATH
# Option B is recommended if you're going to mess with the dependencies.
```

3) Populate your new GOPATH.

```sh
cd $KPATH/src/k8s.io/kubernetes
godep restore
```

4) Next, you can either add a new dependency or update an existing one.

```sh
# To add a new dependency, do:
cd $KPATH/src/k8s.io/kubernetes
go get path/to/dependency
# Change code in Kubernetes to use the dependency.
godep save ./...

# To update an existing dependency, do:
cd $KPATH/src/k8s.io/kubernetes
go get -u path/to/dependency
# Change code in Kubernetes accordingly if necessary.
godep update path/to/dependency/...
```

_If `go get -u path/to/dependency` fails with compilation errors, instead try
`go get -d -u path/to/dependency` to fetch the dependencies without compiling
them. This can happen when updating the cadvisor dependency._


5) Before sending your PR, it's a good idea to sanity check that your
Godeps.json file is ok by running `hack/verify-godeps.sh`

_If hack/verify-godeps.sh fails after a `godep update`, it is possible that a
transitive dependency was added or removed but not updated by godeps. It then
may be necessary to perform a `godep save ./...` to pick up the transitive
dependency changes._

It is sometimes expedient to manually fix the /Godeps/godeps.json file to
minimize the changes.

Please send dependency updates in separate commits within your PR, for easier
reviewing.

6) If you updated the Godeps, please also update `Godeps/LICENSES` by running
`hack/update-godep-licenses.sh`.

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
