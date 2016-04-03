# How to contribute

etcd is Apache 2.0 licensed and accepts contributions via GitHub pull requests. This document outlines some of the conventions on commit message formatting, contact points for developers and other resources to make getting your contribution into etcd easier.

# Email and chat

- Email: [etcd-dev](https://groups.google.com/forum/?hl=en#!forum/etcd-dev)
- IRC: #[coreos](irc://irc.freenode.org:6667/#coreos) IRC channel on freenode.org

## Getting started

- Fork the repository on GitHub
- Read the README.md for build instructions

## Reporting Bugs and Creating Issues

Reporting bugs is one of the best ways to contribute. However, a good bug report
has some very specific qualities, so please read over our short document on
[reporting bugs](https://github.com/coreos/etcd/blob/master/Documentation/reporting_bugs.md)
before you submit your bug report. This document might contain links known
issues, another good reason to take a look there, before reporting your bug.

## Contribution flow

This is a rough outline of what a contributor's workflow looks like:

- Create a topic branch from where you want to base your work. This is usually master.
- Make commits of logical units.
- Make sure your commit messages are in the proper format (see below).
- Push your changes to a topic branch in your fork of the repository.
- Submit a pull request to coreos/etcd.
- Your PR must receive a LGTM from two maintainers found in the MAINTAINERS file.

Thanks for your contributions!

### Code style

The coding style suggested by the Golang community is used in etcd. See the [style doc](https://github.com/golang/go/wiki/CodeReviewComments) for details.

Please follow this style to make etcd easy to review, maintain and develop.

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
