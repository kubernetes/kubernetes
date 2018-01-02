# How to Contribute

CNI is [Apache 2.0 licensed](LICENSE) and accepts contributions via GitHub
pull requests. This document outlines some of the conventions on development
workflow, commit message formatting, contact points and other resources to make
it easier to get your contribution accepted.

We gratefully welcome improvements to documentation as well as to code.

# Certificate of Origin

By contributing to this project you agree to the Developer Certificate of
Origin (DCO). This document was created by the Linux Kernel community and is a
simple statement that you, as a contributor, have the legal right to make the
contribution. See the [DCO](DCO) file for details.

# Email and Chat

The project uses the the cni-dev email list and IRC chat:
- Email: [cni-dev](https://groups.google.com/forum/#!forum/cni-dev)
- IRC: #[containernetworking](irc://irc.freenode.org:6667/#containernetworking) channel on freenode.org

Please avoid emailing maintainers found in the MAINTAINERS file directly. They
are very busy and read the mailing lists.

## Getting Started

- Fork the repository on GitHub
- Read the [README](README.md) for build and test instructions
- Play with the project, submit bugs, submit pull requests!

## Contribution workflow

This is a rough outline of how to prepare a contribution:

- Create a topic branch from where you want to base your work (usually branched from master).
- Make commits of logical units.
- Make sure your commit messages are in the proper format (see below).
- Push your changes to a topic branch in your fork of the repository.
- If you changed code:
   - add automated tests to cover your changes, using the [Ginkgo](http://onsi.github.io/ginkgo/) & [Gomega](http://onsi.github.io/gomega/) style
   - if the package did not previously have any test coverage, add it to the list
   of `TESTABLE` packages in the `test.sh` script.
   - run the full test script and ensure it passes
- Make sure any new code files have a license header (this is now enforced by automated tests)
- Submit a pull request to the original repository.

## How to run the test suite
We generally require test coverage of any new features or bug fixes.

Here's how you can run the test suite on any system (even Mac or Windows) using
 [Vagrant](https://www.vagrantup.com/) and a hypervisor of your choice:

```bash
vagrant up
vagrant ssh
# you're now in a shell in a virtual machine
sudo su
cd /go/src/github.com/containernetworking/cni

# to run the full test suite
./test.sh

# to focus on a particular test suite
cd plugins/main/loopback
go test
```

# Acceptance policy

These things will make a PR more likely to be accepted:

 * a well-described requirement
 * tests for new code
 * tests for old code!
 * new code and tests follow the conventions in old code and tests
 * a good commit message (see below)

In general, we will merge a PR once two maintainers have endorsed it.
Trivial changes (e.g., corrections to spelling) may get waved through.
For substantial changes, more people may become involved, and you might get asked to resubmit the PR or divide the changes into more than one PR.

### Format of the Commit Message

We follow a rough convention for commit messages that is designed to answer two
questions: what changed and why. The subject line should feature the what and
the body of the commit should describe the why.

```
scripts: add the test-cluster command

this uses tmux to setup a test cluster that you can easily kill and
start for debugging.

Fixes #38
```

The format can be described more formally as follows:

```
<subsystem>: <what changed>
<BLANK LINE>
<why this change was made>
<BLANK LINE>
<footer>
```

The first line is the subject and should be no longer than 70 characters, the
second line is always blank, and other lines should be wrapped at 80 characters.
This allows the message to be easier to read on GitHub as well as in various
git tools.

## 3rd party plugins
So you've built a CNI plugin.  Where should it live?

Short answer: We'd be happy to link to it from our [list of 3rd party plugins](README.md#3rd-party-plugins).
But we'd rather you kept the code in your own repo.

Long answer: An advantage of the CNI model is that independent plugins can be
built, distributed and used without any code changes to this repository.  While
some widely used plugins (and a few less-popular legacy ones) live in this repo,
we're reluctant to add more.

If you have a good reason why the CNI maintainers should take custody of your
plugin, please open an issue or PR.
