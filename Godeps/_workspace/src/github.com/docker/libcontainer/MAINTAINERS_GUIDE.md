# The libcontainer Maintainers' Guide

## Introduction

Dear maintainer. Thank you for investing the time and energy to help
make libcontainer as useful as possible. Maintaining a project is difficult,
sometimes unrewarding work.  Sure, you will get to contribute cool
features to the project. But most of your time will be spent reviewing,
cleaning up, documenting, answering questions, justifying design
decisions - while everyone has all the fun! But remember - the quality
of the maintainers work is what distinguishes the good projects from the
great.  So please be proud of your work, even the unglamourous parts,
and encourage a culture of appreciation and respect for *every* aspect
of improving the project - not just the hot new features.

This document is a manual for maintainers old and new. It explains what
is expected of maintainers, how they should work, and what tools are
available to them.

This is a living document - if you see something out of date or missing,
speak up!

## What are a maintainer's responsibility?

It is every maintainer's responsibility to:

* 1) Expose a clear roadmap for improving their component.
* 2) Deliver prompt feedback and decisions on pull requests.
* 3) Be available to anyone with questions, bug reports, criticism etc.
  on their component. This includes IRC, GitHub requests and the mailing
  list.
* 4) Make sure their component respects the philosophy, design and
  roadmap of the project.

## How are decisions made?

Short answer: with pull requests to the libcontainer repository.

libcontainer is an open-source project with an open design philosophy. This
means that the repository is the source of truth for EVERY aspect of the
project, including its philosophy, design, roadmap and APIs. *If it's
part of the project, it's in the repo. It's in the repo, it's part of
the project.*

As a result, all decisions can be expressed as changes to the
repository. An implementation change is a change to the source code. An
API change is a change to the API specification. A philosophy change is
a change to the philosophy manifesto. And so on.

All decisions affecting libcontainer, big and small, follow the same 3 steps:

* Step 1: Open a pull request. Anyone can do this.

* Step 2: Discuss the pull request. Anyone can do this.

* Step 3: Accept (`LGTM`) or refuse a pull request. The relevant maintainers do 
this (see below "Who decides what?")


## Who decides what?

All decisions are pull requests, and the relevant maintainers make
decisions by accepting or refusing the pull request. Review and acceptance
by anyone is denoted by adding a comment in the pull request: `LGTM`. 
However, only currently listed `MAINTAINERS` are counted towards the required
two LGTMs.

libcontainer follows the timeless, highly efficient and totally unfair system
known as [Benevolent dictator for life](http://en.wikipedia.org/wiki/Benevolent_Dictator_for_Life), with Michael Crosby in the role of BDFL.
This means that all decisions are made by default by Michael. Since making
every decision himself would be highly un-scalable, in practice decisions
are spread across multiple maintainers.

The relevant maintainers for a pull request can be worked out in two steps:

* Step 1: Determine the subdirectories affected by the pull request. This
  might be `netlink/` and `security/`, or any other part of the repo.

* Step 2: Find the `MAINTAINERS` file which affects this directory. If the
  directory itself does not have a `MAINTAINERS` file, work your way up
  the repo hierarchy until you find one.

### I'm a maintainer, and I'm going on holiday

Please let your co-maintainers and other contributors know by raising a pull
request that comments out your `MAINTAINERS` file entry using a `#`.

### I'm a maintainer, should I make pull requests too?

Yes. Nobody should ever push to master directly. All changes should be
made through a pull request.

### Who assigns maintainers?

Michael has final `LGTM` approval for all pull requests to `MAINTAINERS` files.

### How is this process changed?

Just like everything else: by making a pull request :)
