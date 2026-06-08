# Project governance

The [OCI charter][charter] §5.b.viii tasks an OCI Project's maintainers (listed in the repository's MAINTAINERS file and sometimes referred to as "the TDC", [§5.e][charter]) with:

> Creating, maintaining and enforcing governance guidelines for the TDC, approved by the maintainers, and which shall be posted visibly for the TDC.

This section describes generic rules and procedures for fulfilling that mandate.

## Proposing a motion

A maintainer SHOULD propose a motion on the dev@opencontainers.org mailing list (except [security issues](#security-issues)) with another maintainer as a co-sponsor.

## Voting

Voting on a proposed motion SHOULD happen on the dev@opencontainers.org mailing list (except [security issues](#security-issues)) with maintainers posting LGTM or REJECT.
Maintainers MAY also explicitly not vote by posting ABSTAIN (which is useful to revert a previous vote).
Maintainers MAY post multiple times (e.g. as they revise their position based on feedback), but only their final post counts in the tally.
A proposed motion is adopted if two-thirds of votes cast, a quorum having voted, are in favor of the release.

Voting SHOULD remain open for a week to collect feedback from the wider community and allow the maintainers to digest the proposed motion.
Under exceptional conditions (e.g. non-major security fix releases) proposals which reach quorum with unanimous support MAY be adopted earlier.

A maintainer MAY choose to reply with REJECT.
A maintainer posting a REJECT MUST include a list of concerns or links to written documentation for those concerns (e.g. GitHub issues or mailing-list threads).
The maintainers SHOULD try to resolve the concerns and wait for the rejecting maintainer to change their opinion to LGTM.
However, a motion MAY be adopted with REJECTs, as outlined in the previous paragraphs.

## Quorum

A quorum is established when at least two-thirds of maintainers have voted.

For projects that are not specifications, a [motion to release](#release-approval) MAY be adopted if the tally is at least three LGTMs and no REJECTs, even if three votes does not meet the usual two-thirds quorum.

## Amendments

The [project governance](#project-governance) rules and procedures MAY be amended or replaced using the procedures themselves.
The MAINTAINERS of this project governance document is the total set of MAINTAINERS from all Open Containers projects (go-digest, image-spec, image-tools, runC, runtime-spec, runtime-tools, and selinux).

## Subject templates

Maintainers are busy and get lots of email.
To make project proposals recognizable, proposed motions SHOULD use the following subject templates.

### Proposing a motion

> [{project} VOTE]: {motion description} (closes {end of voting window})

For example:

> [runtime-spec VOTE]: Tag 0647920 as 1.0.0-rc (closes 2016-06-03 20:00 UTC)

### Tallying results

After voting closes, a maintainer SHOULD post a tally to the motion thread with a subject template like:

> [{project} {status}]: {motion description} (+{LGTMs} -{REJECTs} #{ABSTAINs})

Where `{status}` is either `adopted` or `rejected`.
For example:

> [runtime-spec adopted]: Tag 0647920 as 1.0.0-rc (+6 -0 #3)

[charter]: https://www.opencontainers.org/about/governance
