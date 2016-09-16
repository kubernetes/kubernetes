# Contributing guidelines

Want to hack on kubernetes? Yay!

## Developer Guide

We have a [Developer's Guide](docs/devel/README.md) that outlines everything you need to know from setting up your dev environment to how to get faster Pull Request reviews. If you find something undocumented or incorrect along the way, please feel free to send a Pull Request.

## Filing issues

If you have a question about Kubernetes or have a problem using it, please read the [troubleshooting guide](docs/troubleshooting.md) before filing an issue.

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement (CLA).

  * If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](http://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](http://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository. This policy does not apply to [third_party](third_party/).

### Contributing A Patch

1. Submit an issue describing your proposed change to the repo in question.
1. The repo owner will respond to your issue promptly.
1. If your proposed change is accepted, and you haven't already done so, sign a Contributor License Agreement (see details above).
1. Fork the desired repo, develop and test your code changes.
1. Submit a pull request.

### Protocols for Collaborative Development

Please read [this doc](docs/devel/collab.md) for information on how we're running development for the project.
Also take a look at the [development guide](docs/devel/development.md) for information on how to set up your environment, run tests, manage dependencies, etc.

### Adding dependencies

If your patch depends on new packages, add that package with [`godep`](https://github.com/tools/godep).  Follow the [instructions to add a dependency](docs/devel/development.md#godep-and-dependency-management).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/CONTRIBUTING.md?pixel)]()
