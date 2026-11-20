## Introduction

Dear maintainer. Thank you for investing the time and energy to help
make this project as useful as possible. Maintaining a project is difficult,
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

## What are a maintainer's responsibilities?

It is every maintainer's responsibility to:

* Expose a clear roadmap for improving their component.
* Deliver prompt feedback and decisions on pull requests.
* Be available to anyone with questions, bug reports, criticism etc. on their component.
  This includes IRC and GitHub issues and pull requests.
* Make sure their component respects the philosophy, design and roadmap of the project.

## How are decisions made?

This project is an open-source project with an open design philosophy. This
means that the repository is the source of truth for EVERY aspect of the
project, including its philosophy, design, roadmap and APIs. *If it's
part of the project, it's in the repo. It's in the repo, it's part of
the project.*

As a result, all decisions can be expressed as changes to the
repository. An implementation change is a change to the source code. An
API change is a change to the API specification. A philosophy change is
a change to the philosophy manifesto. And so on.

All decisions affecting this project, big and small, follow the same procedure:

1. Discuss a proposal on the [mailing list](CONTRIBUTING.md#mailing-list).
   Anyone can do this.
2. Open a pull request.
   Anyone can do this.
3. Discuss the pull request.
   Anyone can do this.
4. Endorse (`LGTM`) or oppose (`Rejected`) the pull request.
   The relevant maintainers do this (see below [Who decides what?](#who-decides-what)).
   Changes that affect project management (changing policy, cutting releases, etc.) are [proposed and voted on the mailing list](GOVERNANCE.md).
5. Merge or close the pull request.
   The relevant maintainers do this.

### I'm a maintainer, should I make pull requests too?

Yes. Nobody should ever push to master directly. All changes should be
made through a pull request.

## Who decides what?

All decisions are pull requests, and the relevant maintainers make
decisions by accepting or refusing the pull request. Review and acceptance
by anyone is denoted by approving the pull request (an `LGTM`).
However, only currently listed `MAINTAINERS` are counted towards the required
LGTMs, and the pull request author's own LGTM is never counted.

A pull request can be merged once it has:

* two LGTMs from maintainers; or
* one LGTM from a maintainer, if it is a routine change; or
* one LGTM from another maintainer, if the author is a maintainer, the pull
  request is not a governance change, and it has been open for review (that
  is, not a draft) for at least 7 days, or 14 days if it is a large change
  (more than 100 lines added and removed). This gives other maintainers a
  chance to review and object.

A pull request is a governance change if it changes the rules of the project,
or how they are enforced. That is, it changes `MAINTAINERS`, `CODEOWNERS`,
a top-level `*.md` file other than `README.md` (such as this document), or the
`review-policy` workflow. A governance change always requires two LGTMs.

A pull request is a routine change if it is not a governance change, and it is
either:

* opened by Dependabot; or
* only changes documentation (`*.md` files) and/or CI configuration (files
  under `.github/`).

In all cases, a maintainer requesting changes blocks the merge until the
request is resolved (the reviewer approves, or the review is dismissed).

These rules are enforced by the
[`review-policy`](.github/workflows/review-policy.yml) check.

Overall the maintainer system works because of mutual respect.
The maintainers trust one another to act in the best interests of the project.
Sometimes maintainers can disagree and this is part of a healthy project to represent the points of view of various people.
In the case where maintainers cannot find agreement on a specific change, maintainers should use the [governance procedure](GOVERNANCE.md) to attempt to reach a consensus.

### How are maintainers added?

The best maintainers have a vested interest in the project.  Maintainers
are first and foremost contributors that have shown they are committed to
the long term success of the project.  Contributors wanting to become
maintainers are expected to be deeply involved in contributing code,
pull request review, and triage of issues in the project for more than two months.

Just contributing does not make you a maintainer, it is about building trust with the current maintainers of the project and being a person that they can depend on to act in the best interest of the project.
The final vote to add a new maintainer should be approved by the [governance procedure](GOVERNANCE.md).

### How are maintainers removed?

When a maintainer is unable to perform the [required duties](#what-are-a-maintainers-responsibilities) they can be removed by the [governance procedure](GOVERNANCE.md).
Issues related to a maintainer's performance should be discussed with them among the other maintainers so that they are not surprised by a pull request removing them.
