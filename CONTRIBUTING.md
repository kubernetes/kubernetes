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

Please fill out either the individual or corporate Contributor License
Agreement (CLA).  As of Q4 2016, we're transitioning from Google's CLA to the
Cloud Native Computing Foundation (CNCF) CLA.

Google:
  * If you are an individual writing original source code and you're sure you
    own the intellectual property, then you'll need to sign an [individual
    CLA](http://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work,
    then you'll need to sign a [corporate
    CLA](http://code.google.com/legal/corporate-cla-v1.0.html).

CNCF:
  * To contribute as an individual or as am employee of a signed organization,
    [go here](https://identity.linuxfoundation.org/projects/cncf).
  * To sign up as an organization, [go
    here](https://identity.linuxfoundation.org/node/285/organization-signup).

Once you are CLA'ed, we'll be able to accept your pull requests.

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
