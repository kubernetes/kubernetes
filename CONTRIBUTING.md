# Contributing guidelines

Want to hack on Kubernetes? Yay!

## Developer Guide

We have a [Developer's Guide](docs/devel/README.md) that outlines everything
you need to know from setting up your dev environment to how to get faster Pull
Request reviews. If you find something undocumented or incorrect along the way,
please feel free to send a Pull Request.

## Filing issues

If you have a question about Kubernetes or have a problem using it, please
start with the [troubleshooting guide](http://kubernetes.io/docs/troubleshooting/).  If that
doesn't answer your questions, or if you think you found a bug, please [file an
issue](https://github.com/kubernetes/kubernetes/issues/new).

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a
couple of legal hurdles.

The Cloud Native Computing Foundation (CNCF) CLA [must be signed](https://github.com/kubernetes/community/blob/master/CLA.md) by all contributors.
Please fill out either the individual or corporate Contributor License
Agreement (CLA).

Once you are CLA'ed, we'll be able to accept your pull requests. For any issues that you face during this process,
please add a comment [here](https://github.com/kubernetes/kubernetes/issues/27796) explaining the issue and we will help get it sorted out.

***NOTE***: Only original source code from you and other people that have
signed the CLA can be accepted into the repository. This policy does not
apply to [third_party](third_party/) and [vendor](vendor/).

### Finding Things That Need Help

If you're new to the project and want to help, but don't know where to start,
we have a semi-curated list of issues that have should not need deep knowledge
of the system.  [Have a look and see if anything sounds
interesting](https://github.com/kubernetes/kubernetes/issues?q=is%3Aopen+is%3Aissue+label%3Ahelp-wanted).

Alternatively, read some of the many docs on the system, for example [the
architecture](docs/design/architecture.md), and pick a component that seems
interesting.  Start with `main()` (look in the [cmd](cmd/) directory) and read
until you find something you want to fix.  The best way to learn is to hack!
There's always code that can be clarified and variables or functions that can
be renamed or commented.

### Contributing A Patch

If you're working on an existing issue, such as one of the `help-wanted` ones
above, simply respond to the issue and express interest in working on it.  This
helps other people know that the issue is active, and hopefully prevents
duplicated efforts.

If you want to work on a new idea of relatively small scope:

1. Submit an issue describing your proposed change to the repo in question.
1. The repo owners will respond to your issue promptly.
1. If your proposed change is accepted, and you haven't already done so, sign a
   Contributor License Agreement (see details above).
1. Fork the repo, develop, and test your changes.
1. Submit a pull request.

If you want to work on a bigger idea, we STRONGLY recommend that you start with
some bugs or smaller features.  We have a [feature development
process](https://github.com/kubernetes/features/blob/master/README.md), but
navigating the Kubernetes system as a newcomer can be very challenging.

### Downloading the project

There are a few ways you can download this code.  You must download it into a
GOPATH - see [golang.org](https://golang.org/doc/code.html) for more info on
how Go works with code.  This project expects to be found at the Go package
`k8s.io/kubernetes`.

1. You can `git clone` the repo.  If you do this, you MUST make sure it is in
   the GOPATH as `k8s.io/kubernetes` or it may not build.  E.g.: `git clone
   https://github.com/kubernetes/kubernetes $GOPATH/src/k8s.io/kubernetes`
1. You can use `go get` to fetch the repo.  This will automatically put it into
   your GOPATH in the right place. E.g.: `go get -d k8s.io/kubernetes`
1. You can download an archive of the source.  If you do this, you MUST make
   sure it is unpacked into the GOPATH as `k8s.io/kubernetes` or it may not
   build. See [rel.k8s.io](http://rel.k8s.io) for a list of available releases.

### Building the project

There are a few things you need to build and test this project:

1. `make` - the human interface to the Kubernetes build is `make`, so you must
   have this tool installed on your machine.  We try not to use too many crazy
   features of `Makefile`s and other tools, so most commonly available versions
   should work.
1. `docker` - some parts of the build/test system depend on `docker`.  You
   need a relatively recent version of Docker installed, and available to you.
1. `go` - Kubernetes is written in Go (aka golang), so you need a relatively
   recent version of the [Go toolchain](https://golang.org/dl/) installed.
   While Linux is the primary platform for Kubernetes, it should compile on a
   Mac, too.  Windows is in progress.

To build Kubernetes, simply type `make`.  This should figure out what it needs
to do and not need any input from you.  If you want to just build a subset of
code, you can pass the `WHAT` variable to `make`: e.g. `make
WHAT="cmd/kubelet"`.

To run basic tests, simply type `make test`.  This will run all of the unit
tests in the project.  If you want to just test a subset of the project, you
can pass the `WHAT` variable to `make`: e.g. `make test WHAT=pkg/kubelet`.

### Protocols for Collaborative Development

Please read [this doc](docs/devel/collab.md) for information on how we're
running development for the project.  Also take a look at the [development
guide](docs/devel/development.md) for information on how to set up your
environment, run tests, manage dependencies, etc.

### Adding dependencies

If your patch depends on new packages, add that package with
[`godep`](https://github.com/tools/godep).  Follow the [instructions to add a
dependency](docs/devel/development.md#godep-and-dependency-management).

### Community Expectations

Please see our [expectations](docs/devel/community-expectations.md) for members
of the Kubernetes community.



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/CONTRIBUTING.md?pixel)]()
