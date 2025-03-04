# Releases

The release process hopes to encourage early, consistent consensus-building during project development.
The mechanisms used are regular community communication on the mailing list about progress, scheduled meetings for issue resolution and release triage, and regularly paced and communicated releases.
Releases are proposed and adopted or rejected using the usual [project governance](GOVERNANCE.md) rules and procedures.

An anti-pattern that we want to avoid is heavy development or discussions "late cycle" around major releases.
We want to build a community that is involved and communicates consistently through all releases instead of relying on "silent periods" as a judge of stability.

## Parallel releases

A single project MAY consider several motions to release in parallel.
However each motion to release after the initial 0.1.0 MUST be based on a previous release that has already landed.

For example, runtime-spec maintainers may propose a v1.0.0-rc2 on the 1st of the month and a v0.9.1 bugfix on the 2nd of the month.
They may not propose a v1.0.0-rc3 until the v1.0.0-rc2 is accepted (on the 7th if the vote initiated on the 1st passes).

## Specifications

The OCI maintains three categories of projects: specifications, applications, and conformance-testing tools.
However, specification releases have special restrictions in the [OCI charter][charter]:

* They are the target of backwards compatibility (§7.g), and
* They are subject to the OFWa patent grant (§8.d and e).

To avoid unfortunate side effects (onerous backwards compatibity requirements or Member resignations), the following additional procedures apply to specification releases:

### Planning a release

Every OCI specification project SHOULD hold meetings that involve maintainers reviewing pull requests, debating outstanding issues, and planning releases.
This meeting MUST be advertised on the project README and MAY happen on a phone call, video conference, or on IRC.
Maintainers MUST send updates to the dev@opencontainers.org with results of these meetings.

Before the specification reaches v1.0.0, the meetings SHOULD be weekly.
Once a specification has reached v1.0.0, the maintainers may alter the cadence, but a meeting MUST be held within four weeks of the previous meeting.

The release plans, corresponding milestones and estimated due dates MUST be published on GitHub (e.g. https://github.com/opencontainers/runtime-spec/milestones).
GitHub milestones and issues are only used for community organization and all releases MUST follow the [project governance](GOVERNANCE.md) rules and procedures.

### Timelines

Specifications have a variety of different timelines in their lifecycle.

* Pre-v1.0.0 specifications SHOULD release on a monthly cadence to garner feedback.
* Major specification releases MUST release at least three release candidates spaced a minimum of one week apart.
  This means a major release like a v1.0.0 or v2.0.0 release will take 1 month at minimum: one week for rc1, one week for rc2, one week for rc3, and one week for the major release itself.
  Maintainers SHOULD strive to make zero breaking changes during this cycle of release candidates and SHOULD restart the three-candidate count when a breaking change is introduced.
  For example if a breaking change is introduced in v1.0.0-rc2 then the series would end with v1.0.0-rc4 and v1.0.0.
* Minor and patch releases SHOULD be made on an as-needed basis.

[charter]: https://www.opencontainers.org/about/governance
