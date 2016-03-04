<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Development Guide

This document is intended to be the canonical source of truth for things like
supported toolchain versions for building Kubernetes.  If you find a
requirement that this doc does not capture, please file a bug.  If you find
other docs with references to requirements that are not simply links to this
doc, please file a bug.

This document is intended to be relative to the branch in which it is found.
It is guaranteed that requirements will change over time for the development
branch, but release branches of Kubernetes should not change.

## Releases and Official Builds

Official releases are built in Docker containers.  Details are [here](http://releases.k8s.io/release-1.2/build/README.md).  You can do simple builds and development with just a local Docker installation.  If you want to build go code locally outside of docker, please continue below.

## Go development environment

Kubernetes is written in the [Go](http://golang.org) programming language. If you haven't set up a Go development environment, please follow [these instructions](http://golang.org/doc/code.html) to install the go tools and set up a GOPATH.

### Go versions

Requires Go version 1.4.x or 1.5.x

## Git setup

Below, we outline one of the more common git workflows that core developers use. Other git workflows are also valid.

### Visual overview

![Git workflow](git_workflow.png)

### Fork the main repository

1. Go to https://github.com/kubernetes/kubernetes
2. Click the "Fork" button (at the top right)

### Clone your fork

The commands below require that you have $GOPATH set ([$GOPATH docs](https://golang.org/doc/code.html#GOPATH)). We highly recommend you put Kubernetes' code into your GOPATH. Note: the commands below will not work if there is more than one directory in your `$GOPATH`.

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
directory. This will keep you from accidentally committing non-gofmt'd go code.

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
sorts of commits.  It is not imperative that every commit in a PR compile and
pass tests independently, but it is worth striving for.  For mass automated
fixups (e.g. automated doc formatting), use one or more commits for the
changes to tooling and a final commit to apply the fixup en masse.  This makes
reviews much easier.

See [Faster Reviews](faster_reviews.md) for more details.

## godep and dependency management

Kubernetes uses [godep](https://github.com/tools/godep) to manage dependencies. It is not strictly required for building Kubernetes but it is required when managing dependencies under the Godeps/ tree, and is required by a number of the build and test scripts. Please make sure that ``godep`` is installed and in your ``$PATH``.

### Installing godep

There are many ways to build and host go binaries. Here is an easy way to get utilities like `godep` installed:

1) Ensure that [mercurial](http://mercurial.selenic.com/wiki/Download) is installed on your system. (some of godep's dependencies use the mercurial
source control system).  Use `apt-get install mercurial` or `yum install mercurial` on Linux, or [brew.sh](http://brew.sh) on OS X, or download
directly from mercurial.

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

### Using godep

Here's a quick walkthrough of one way to use godeps to add or update a Kubernetes dependency into Godeps/_workspace. For more details, please see the instructions in [godep's documentation](https://github.com/tools/godep).

1) Devote a directory to this endeavor:

_Devoting a separate directory is not required, but it is helpful to separate dependency updates from other changes._

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

_If `go get -u path/to/dependency` fails with compilation errors, instead try `go get -d -u path/to/dependency`
to fetch the dependencies without compiling them.  This can happen when updating the cadvisor dependency._


5) Before sending your PR, it's a good idea to sanity check that your Godeps.json file is ok by running `hack/verify-godeps.sh`

_If hack/verify-godeps.sh fails after a `godep update`, it is possible that a transitive dependency was added or removed but not
updated by godeps.  It then may be necessary to perform a `godep save ./...` to pick up the transitive dependency changes._

It is sometimes expedient to manually fix the /Godeps/godeps.json file to minimize the changes.

Please send dependency updates in separate commits within your PR, for easier reviewing.

6) If you updated the Godeps, please also update `Godeps/LICENSES` by running `hack/update-godep-licenses.sh`.


## Unit tests

```sh
cd kubernetes
hack/test-go.sh
```

Alternatively, you could also run:

```sh
cd kubernetes
godep go test ./...
```

If you only want to run unit tests in one package, you could run ``godep go test`` under the package directory. For example, the following commands will run all unit tests in package kubelet:

```console
$ cd kubernetes # step into the kubernetes directory.
$ cd pkg/kubelet
$ godep go test
# some output from unit tests
PASS
ok      k8s.io/kubernetes/pkg/kubelet   0.317s
```

## Coverage

Currently, collecting coverage is only supported for the Go unit tests.

To run all unit tests and generate an HTML coverage report, run the following:

```sh
cd kubernetes
KUBE_COVER=y hack/test-go.sh
```

At the end of the run, an the HTML report will be generated with the path printed to stdout.

To run tests and collect coverage in only one package, pass its relative path under the `kubernetes` directory as an argument, for example:

```sh
cd kubernetes
KUBE_COVER=y hack/test-go.sh pkg/kubectl
```

Multiple arguments can be passed, in which case the coverage results will be combined for all tests run.

Coverage results for the project can also be viewed on [Coveralls](https://coveralls.io/r/kubernetes/kubernetes), and are continuously updated as commits are merged. Additionally, all pull requests which spawn a Travis build will report unit test coverage results to Coveralls. Coverage reports from before the Kubernetes Github organization was created can be found [here](https://coveralls.io/r/GoogleCloudPlatform/kubernetes).

## Integration tests

You need an [etcd](https://github.com/coreos/etcd/releases) in your path. To download a copy of the latest version used by Kubernetes, either
 * run `hack/install-etcd.sh`, which will download etcd to `third_party/etcd`, and then set your `PATH` to include `third_party/etcd`.
 * inspect `cluster/saltbase/salt/etcd/etcd.manifest` for the correct version, and then manually download and install it to some place in your `PATH`.

```sh
cd kubernetes
hack/test-integration.sh
```

## End-to-End tests

See [End-to-End Testing in Kubernetes](e2e-tests.md).

## Testing out flaky tests

[Instructions here](flaky-tests.md)

## Benchmarking

To run benchmark tests, you'll typically use something like:

    $ godep go test ./pkg/apiserver -benchmem -run=XXX -bench=BenchmarkWatch

The `-run=XXX` prevents normal unit tests for running, while `-bench` is a regexp for selecting which benchmarks to run.
See `go test -h` for more instructions on generating profiles from benchmarks.

## Regenerating the CLI documentation

```sh
hack/update-generated-docs.sh
```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/development.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
