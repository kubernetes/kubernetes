# How to Contribute

CoreOS projects are [Apache 2.0 licensed](LICENSE) and accept contributions via
GitHub pull requests.  This document outlines some of the conventions on
development workflow, commit message formatting, contact points and other
resources to make it easier to get your contribution accepted.

### Certificate of Origin

By contributing to this project you agree to the Developer Certificate of
Origin (DCO). This document was created by the Linux Kernel community and is a
simple statement that you, as a contributor, have the legal right to make the
contribution. See the [DCO](DCO) file for details.

### Email and Chat

The project has a mailing list and two discussion channels in IRC:
- Email: [rkt-dev](https://groups.google.com/forum/#!forum/rkt-dev)
- IRC: #[rkt](irc://irc.freenode.org:6667/#rkt) on freenode.org, for general discussion
- IRC: #[rkt-dev](irc://irc.freenode.org:6667/#rkt-dev) on freenode.org, for development discussion

Please avoid emailing maintainers found in the MAINTAINERS file directly. They
are very busy and read the mailing lists.

### Getting Started

- Fork the repository on GitHub
- Read [`building rkt`](Documentation/hacking.md#building-rkt) for build and [`manually-running-the-tests`](tests/README.md#manually-running-the-tests) for test instructions
- Play with the project, submit bugs, submit patches!

### Contribution Flow

This is a rough outline of what a contributor's workflow looks like:

- Create a topic branch from where you want to base your work (usually master).
- Make commits of logical units.
- Make sure your commit messages are in the proper format (see below).
- Push your changes to a topic branch in your fork of the repository.
- Make sure the [tests](tests/README.md#manually-running-the-tests) pass, and add any new tests as appropriate.
- Submit a pull request to the original repository.
- Submit a comment with the sole content "@reviewer PTAL" (please take a look) in GitHub
  and replace "@reviewer" with the correct recipient.
- When addressing pull request review comments add new commits to the existing pull request or,
  if the added commits are about the same size as the previous commits,
  squash them into the existing commits.
- Once your PR is labelled as "reviewed/lgtm" squash the addressed commits in one commit.
- If your PR addresses multiple subsystems reorganize your PR and create multiple commits per subsystem.
- Your contribution is ready to be merged.

Thanks for your contributions!

### Coding Style

CoreOS projects written in Go follow a set of style guidelines that we've documented
[here](https://github.com/coreos/docs/tree/master/golang). Please follow them when
working on your contributions.

### Documentation Style

CoreOS project docs should follow the [Documentation style and formatting
guide](https://github.com/coreos/docs/tree/master/STYLE.md). Thank you for documenting!

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

### Format of the Pull Request

The pull request title and the first paragraph of the pull request description
is being used to generate the changelog of the next release.

The convention follows the same rules as for commit messages. The PR title reflects the
what and the first paragraph of the PR description reflects the why.
In most cases one can reuse the commit title as the PR title
and the commit messages as the PR description for the PR.

If your PR includes more commits spanning mulitple subsystems one should change the PR title
and the first paragraph of the PR description to reflect a summary of all changes involved.

A large PR must be split into multiple commits, each with clear commit messages.
Intermediate commits should compile and pass tests. Exceptions to non-compilable must have a valid reason, i.e. dependency bumps.

Do not add entries in the changelog yourself. They will be overwritten when creating a new release.
