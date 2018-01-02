# Docker Release Process

This document describes how the Docker project is released. The Docker project
release process targets the Engine, Compose, Kitematic, Machine, Swarm,
Distribution, Notary and their underlying dependencies (libnetwork, libkv,
etc...).

Step-by-step technical details of the process are described in 
[RELEASE-CHECKLIST.md](https://github.com/docker/docker/blob/master/project/RELEASE-CHECKLIST.md).

## Release cycle

The Docker project follows a **time-based release cycle** and ships every nine
weeks. A release cycle starts the same day the previous release cycle ends.

The first six weeks of the cycle are dedicated to development and review. During
this phase, new features and bugfixes submitted to any of the projects are
**eligible** to be shipped as part of the next release. No changeset submitted
during this period is however guaranteed to be merged for the current release
cycle.

## The freeze period

Six weeks after the beginning of the cycle, the codebase is officially frozen
and the codebase reaches a state close to the final release. A Release Candidate
(RC) gets created at the same time. The freeze period is used to find bugs and
get feedback on the state of the RC before the release.

During this freeze period, while the `master` branch will continue its normal
development cycle, no new features are accepted into the RC. As bugs are fixed
in `master` the release owner will selectively 'cherry-pick' critical ones to
be included into the RC. As the RC changes, new ones are made available for the
community to test and review.

This period lasts for three weeks.

## How to maximize chances of being merged before the freeze date?

First of all, there is never a guarantee that a specific changeset is going to
be merged. However there are different actions to follow to maximize the chances
for a changeset to be merged:

- The team gives priority to review the PRs aligned with the Roadmap (usually
defined by a ROADMAP.md file at the root of the repository).
- The earlier a PR is opened, the more time the maintainers have to review. For
example, if a PR is opened the day before the freeze date, it’s very unlikely
that it will be merged for the release.
- Constant communication with the maintainers (mailing-list, IRC, Github issues,
etc.) allows to get early feedback on the design before getting into the
implementation, which usually reduces the time needed to discuss a changeset.
- If the code is commented, fully tested and by extension follows every single
rules defined by the [CONTRIBUTING guide](
https://github.com/docker/docker/blob/master/CONTRIBUTING.md), this will help
the maintainers by speeding up the review.

## The release

At the end of the freeze (nine weeks after the start of the cycle), all the
projects are released together.

```
                                        Codebase              Release
Start of                                is frozen             (end of the
the Cycle                               (7th week)            9th week)
+---------------------------------------+---------------------+
|                                       |                     |
|           Development phase           |    Freeze phase     |
|                                       |                     |
+---------------------------------------+---------------------+
                   6 weeks                      3 weeks
<---------------------------------------><-------------------->
```

## Exceptions

If a critical issue is found at the end of the freeze period and more time is
needed to address it, the release will be pushed back. When a release gets
pushed back, the next release cycle gets delayed as well.
