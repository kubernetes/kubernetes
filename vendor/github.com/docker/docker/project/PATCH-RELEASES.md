# Docker patch (bugfix) release process

Patch releases (the 'Z' in vX.Y.Z) are intended to fix major issues in a
release. Docker open source projects follow these procedures when creating a
patch release;

After each release (both "major" (vX.Y.0) and "patch" releases (vX.Y.Z)), a
patch release milestone (vX.Y.Z + 1) is created.

The creation of a patch release milestone is no obligation to actually
*create* a patch release. The purpose of these milestones is to collect
issues and pull requests that can *justify* a patch release;

- Any maintainer is allowed to add issues and PR's to the milestone, when
  doing so, preferably leave a comment on the issue or PR explaining *why*
  you think it should be considered for inclusion in a patch release.
- Issues introduced in version vX.Y.0 get added to milestone X.Y.Z+1
- Only *regressions* should be added. Issues *discovered* in version vX.Y.0,
  but already present in version vX.Y-1.Z should not be added, unless
  critical.
- Patch releases can *only* contain bug-fixes. New features should
  *never* be added to a patch release.

The release captain of the "major" (X.Y.0) release, is also responsible for
patch releases. The release captain, together with another maintainer, will
review issues and PRs on the milestone, and assigns `priority/`labels. These
review sessions take place on a weekly basis, more frequent if needed:

- A P0 priority is assigned to critical issues. A maintainer *must* be
  assigned to these issues. Maintainers should strive to fix a P0 within a week.
- A P1 priority is assigned to major issues, but not critical. A maintainer
  *must* be assigned to these issues.
- P2 and P3 priorities are assigned to other issues. A maintainer can be
  assigned.
- Non-critical issues and PR's can be removed from the milestone. Minor
  changes, such as typo-fixes or omissions in the documentation can be
  considered for inclusion in a patch release.

## Deciding if a patch release should be done

- Only a P0 can justify to proceed with the patch release.
- P1, P2, and P3 issues/PR's should not influence the decision, and
  should be moved to the X.Y.Z+1 milestone, or removed from the
  milestone.

> **Note**: If the next "major" release is imminent, the release captain
> can decide to cancel a patch release, and include the patches in the
> upcoming major release.

> **Note**: Security releases are also "patch releases", but follow
> a different procedure. Security releases are developed in a private
> repository, released and tested under embargo before they become
> publicly available.

## Deciding on the content of a patch release

When the criteria for moving forward with a patch release are met, the release
manager will decide on the exact content of the release.

- Fixes to all P0 issues *must* be included in the release.
- Fixes to *some* P1, P2, and P3 issues *may* be included as part of the patch
  release depending on the severity of the issue and the risk associated with
  the patch.

Any code delivered as part of a patch release should make life easier for a
significant amount of users with zero chance of degrading anybody's experience.
A good rule of thumb for that is to limit cherry-picking to small patches, which
fix well-understood issues, and which come with verifiable tests.
