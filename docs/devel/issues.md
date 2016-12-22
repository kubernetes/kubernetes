## GitHub Issues for the Kubernetes Project

A quick overview of how we will review and prioritize incoming issues at
https://github.com/kubernetes/kubernetes/issues

### Priorities

We use GitHub issue labels for prioritization. The absence of a priority label
means the bug has not been reviewed and prioritized yet.

We try to apply these priority labels consistently across the entire project,
but if you notice an issue that you believe to be incorrectly prioritized,
please do let us know and we will evaluate your counter-proposal.

- **priority/P0**: Must be actively worked on as someone's top priority right
now. Stuff is burning. If it's not being actively worked on, someone is expected
to drop what they're doing immediately to work on it. Team leaders are
responsible for making sure that all P0's in their area are being actively
worked on. Examples include user-visible bugs in core features, broken builds or
tests and critical security issues.

- **priority/P1**: Must be staffed and worked on either currently, or very soon,
ideally in time for the next release.

- **priority/P2**: There appears to be general agreement that this would be good
to have, but we may not have anyone available to work on it right now or in the
immediate future. Community contributions would be most welcome in the mean time
(although it might take a while to get them reviewed if reviewers are fully
occupied with higher priority issues, for example immediately before a release).

- **priority/P3**: Possibly useful, but not yet enough support to actually get
it done. These are mostly place-holders for potentially good ideas, so that they
don't get completely forgotten, and can be referenced/deduped every time they
come up.

### Milestones

We additionally use milestones, based on minor version, for determining if a bug
should be fixed for the next release. These milestones will be especially
scrutinized as we get to the weeks just before a release. We can release a new
version of Kubernetes once they are empty. We will have two milestones per minor
release.

- **vX.Y**: The list of bugs that will be merged for that milestone once ready.

- **vX.Y-candidate**: The list of bug that we might merge for that milestone. A
bug shouldn't be in this milestone for more than a day or two towards the end of
a milestone. It should be triaged either into vX.Y, or moved out of the release
milestones.

The above priority scheme still applies. P0 and P1 issues are work we feel must
get done before release.  P2 and P3 issues are work we would merge into the
release if it gets done, but we wouldn't block the release on it. A few days
before release, we will probably move all P2 and P3 bugs out of that milestone
in bulk.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/issues.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
