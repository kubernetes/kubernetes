# Development Guide

# Releases and Official Builds

Official releases are built in Docker containers.  Details are [here](../../build/README.md).  You can do simple builds and development with just a local Docker installation.  If want to build go locally outside of docker, please continue below.

## Go development environment

Kubernetes is written in [Go](http://golang.org) programming language. If you haven't set up Go development environment, please follow [this instruction](http://golang.org/doc/code.html) to install go tool and set up GOPATH. Ensure your version of Go is at least 1.3.

## Put kubernetes into GOPATH

We highly recommend to put kubernetes' code into your GOPATH. For example, the following commands will download kubernetes' code under the current user's GOPATH (Assuming there's only one directory in GOPATH.):

```
$ echo $GOPATH
/home/user/goproj
$ mkdir -p $GOPATH/src/github.com/GoogleCloudPlatform/
$ cd $GOPATH/src/github.com/GoogleCloudPlatform/
$ git clone https://github.com/GoogleCloudPlatform/kubernetes.git
```

The commands above will not work if there are more than one directory in ``$GOPATH``.

(Obviously, clone your own fork of Kubernetes if you plan to do development.)

## godep and dependency management

Kubernetes uses [godep](https://github.com/tools/godep) to manage dependencies. It is not strictly required for building Kubernetes but it is required when managing dependencies under the Godeps/ tree, and is required by a number of the build and test scripts. Please make sure that ``godep`` is installed and in your ``$PATH``.

### Installing godep
There are many ways to build and host go binaries. Here is an easy way to get utilities like ```godep``` installed:

1. Ensure that [mercurial](http://mercurial.selenic.com/wiki/Download) is installed on your system. (some of godep's dependencies use the mercurial
source control system).  Use ```apt-get install mercurial``` or ```yum install mercurial``` on Linux, or [brew.sh](http://brew.sh) on OS X, or download
directly from mercurial.
2. Create a new GOPATH for your tools and install godep:
```
export GOPATH=$HOME/go-tools
mkdir -p $GOPATH
go get github.com/tools/godep
```

3. Add $GOPATH/bin to your path. Typically you'd add this to your ~/.profile:
```
export GOPATH=$HOME/go-tools
export PATH=$PATH:$GOPATH/bin
```

### Using godep
Here is a quick summary of `godep`.  `godep` helps manage third party dependencies by copying known versions into Godeps/_workspace.  You can use `godep` in three ways:

1. Use `godep` to call your `go` commands.  For example: `godep go test ./...`
2. Use `godep` to modify your `$GOPATH` so that other tools know where to find the dependencies.  Specifically: `export GOPATH=$GOPATH:$(godep path)`
3. Use `godep` to copy the saved versions of packages into your `$GOPATH`.  This is done with `godep restore`.

We recommend using options #1 or #2.

## Hooks

Before committing any changes, please link/copy these hooks into your .git
directory. This will keep you from accidentally committing non-gofmt'd go code.

```
cd kubernetes/.git/hooks/
ln -s ../../hooks/prepare-commit-msg .
ln -s ../../hooks/commit-msg .
```

## Unit tests

```
cd kubernetes
hack/test-go.sh
```

Alternatively, you could also run:

```
cd kubernetes
godep go test ./...
```

If you only want to run unit tests in one package, you could run ``godep go test`` under the package directory. For example, the following commands will run all unit tests in package kubelet:

```
$ cd kubernetes # step into kubernetes' directory.
$ cd pkg/kubelet
$ godep go test
# some output from unit tests
PASS
ok      github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet   0.317s
```

## Coverage
```
cd kubernetes
godep go tool cover -html=target/c.out
```

## Integration tests

You need an etcd somewhere in your PATH. To install etcd, run:

```
cd kubernetes
hack/travis/install-etcd.sh
sudo ln -s $(pwd)/third_party/etcd/bin/etcd /usr/bin/etcd
```

```
cd kubernetes
hack/test-integration.sh
```

## End-to-End tests

You can run an end-to-end test which will bring up a master and two minions, perform some tests, and then tear everything down. Make sure you have followed the getting started steps for your chosen cloud platform (which might involve changing the `KUBERNETES_PROVIDER` environment variable to something other than "gce".
```
cd kubernetes
hack/e2e-test.sh
```

Pressing control-C should result in an orderly shutdown but if something goes wrong and you still have some VMs running you can force a cleanup with the magical incantation:
```
hack/e2e-test.sh 1 1 1
```

## Testing out flaky tests
[Instructions here](docs/devel/flaky-tests.md)

## Add/Update dependencies

Kubernetes uses [godep](https://github.com/tools/godep) to manage dependencies. To add or update a package, please follow the instructions on [godep's document](https://github.com/tools/godep).

To add a new package ``foo/bar``:

- Make sure the kubernetes' root directory is in $GOPATH/github.com/GoogleCloudPlatform/kubernetes
- Run ``godep restore`` to make sure you have all dependancies pulled.
- Download foo/bar into the first directory in GOPATH: ``go get foo/bar``.
- Change code in kubernetes to use ``foo/bar``.
- Run ``godep save ./...`` under kubernetes' root directory.

To update a package ``foo/bar``:

- Make sure the kubernetes' root directory is in $GOPATH/github.com/GoogleCloudPlatform/kubernetes
- Run ``godep restore`` to make sure you have all dependancies pulled.
- Update the package with ``go get -u foo/bar``.
- Change code in kubernetes accordingly if necessary.
- Run ``godep update foo/bar`` under kubernetes' root directory.

## Keeping your development fork in sync

One time after cloning your forked repo:

```
git remote add upstream https://github.com/GoogleCloudPlatform/kubernetes.git
```

Then each time you want to sync to upstream:

```
git fetch upstream
git rebase upstream/master
```

## Regenerating the API documentation

```
cd kubernetes/api
sudo docker build -t kubernetes/raml2html .
sudo docker run --name="docgen" kubernetes/raml2html
sudo docker cp docgen:/data/kubernetes.html .
```

View the API documentation using htmlpreview (works on your fork, too):
```
http://htmlpreview.github.io/?https://github.com/GoogleCloudPlatform/kubernetes/blob/master/api/kubernetes.html
```
