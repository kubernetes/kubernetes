# Pull request reviewing process

## Labels

Labels are carefully picked to optimize for:

 - Readability: maintainers must immediately know the state of a PR
 - Filtering simplicity: different labels represent many different aspects of
   the reviewing work, and can even be targeted at different maintainers groups.

A pull request should only be attributed labels documented in this section: other labels that may
exist on the repository should apply to issues.

### DCO labels

 * `dco/no`: automatically set by a bot when one of the commits lacks proper signature

### Status labels

 * `status/0-triage`
 * `status/1-design-review`
 * `status/2-code-review`
 * `status/3-docs-review`
 * `status/4-ready-to-merge`

Special status labels:

 * `status/failing-ci`: indicates that the PR in its current state fails the test suite
 * `status/needs-attention`: calls for a collective discussion during a review session

### Impact labels (apply to merged pull requests)

 * `impact/api`
 * `impact/changelog`
 * `impact/cli`
 * `impact/deprecation`
 * `impact/distribution`
 * `impact/dockerfile`

### Process labels (apply to merged pull requests)

Process labels are to assist in preparing (patch) releases. These labels should only be used for pull requests.

Label                           | Use for
------------------------------- | -------------------------------------------------------------------------
`process/cherry-pick`           | PRs that should be cherry-picked in the bump/release branch. These pull-requests must also be assigned to a milestone.
`process/cherry-picked`         | PRs that have been cherry-picked. This label is helpful to find PR's that have been added to release-candidates, and to update the change log
`process/docs-cherry-pick`      | PRs that should be cherry-picked in the docs branch. Only apply this label for changes that apply to the *current* release, and generic documentation fixes, such as Markdown and spelling fixes.
`process/docs-cherry-picked`    | PRs that have been cherry-picked in the docs branch
`process/merge-to-master`       | PRs that are opened directly on the bump/release branch, but also need to be merged back to "master"
`process/merged-to-master`      | PRs that have been merged back to "master"


## Workflow

An opened pull request can be in 1 of 5 distinct states, for each of which there is a corresponding
label that needs to be applied.

### Triage - `status/0-triage`

Maintainers are expected to triage new incoming pull requests by removing the `status/0-triage`
label and adding the correct labels (e.g. `status/1-design-review`) before any other interaction
with the PR. The starting label may potentially skip some steps depending on the kind of pull
request: use your best judgement.

Maintainers should perform an initial, high-level, overview of the pull request before moving it to
the next appropriate stage:

 - Has DCO
 - Contains sufficient justification (e.g., usecases) for the proposed change
 - References the Github issue it fixes (if any) in the commit or the first Github comment

Possible transitions from this state:

 * Close: e.g., unresponsive contributor without DCO
 * `status/1-design-review`: general case
 * `status/2-code-review`: e.g. trivial bugfix
 * `status/3-docs-review`: non-proposal documentation-only change

### Design review - `status/1-design-review`

Maintainers are expected to comment on the design of the pull request.  Review of documentation is
expected only in the context of design validation, not for stylistic changes.

Ideally, documentation should reflect the expected behavior of the code.  No code review should
take place in this step.

There are no strict rules on the way a design is validated: we usually aim for a consensus,
although a single maintainer approval is often sufficient for obviously reasonable changes. In
general, strong disagreement expressed by any of the maintainers should not be taken lightly.

Once design is approved, a maintainer should make sure to remove this label and add the next one.

Possible transitions from this state:

 * Close: design rejected
 * `status/2-code-review`: general case
 * `status/3-docs-review`: proposals with only documentation changes

### Code review - `status/2-code-review`

Maintainers are expected to review the code and ensure that it is good quality and in accordance
with the documentation in the PR.

New testcases are expected to be added. Ideally, those testcases should fail when the new code is
absent, and pass when present. The testcases should strive to test as many variants, code paths, as
possible to ensure maximum coverage.

Changes to code must be reviewed and approved (LGTM'd) by a minimum of two code maintainers. When
the author of a PR is a maintainer, he still needs the approval of two other maintainers.

Once code is approved according to the rules of the subsystem, a maintainer should make sure to
remove this label and add the next one. If documentation is absent but expected, maintainers should
ask for documentation and move to status `status/3-docs-review` for docs maintainer to follow.

Possible transitions from this state:

 * Close
 * `status/1-design-review`: new design concerns are raised
 * `status/3-docs-review`: general case
 * `status/4-ready-to-merge`: change not impacting documentation

### Docs review - `status/3-docs-review`

Maintainers are expected to review the documentation in its bigger context, ensuring consistency,
completeness, validity, and breadth of coverage across all existing and new documentation.

They should ask for any editorial change that makes the documentation more consistent and easier to
understand.

The docker/docker repository only contains _reference documentation_, all
"narrative" documentation is kept in a [unified documentation
repository](https://github.com/docker/docker.github.io). Reviewers must
therefore verify which parts of the documentation need to be updated. Any
contribution that may require changing the narrative should get the
`impact/documentation` label: this is the signal for documentation maintainers
that a change will likely need to happen on the unified documentation
repository. When in doubt, it’s better to add the label and leave it to
documentation maintainers to decide whether it’s ok to skip. In all cases,
leave a comment to explain what documentation changes you think might be needed.

- If the pull request does not impact the documentation at all, the docs review
  step is skipped, and the pull request is ready to merge.
- If the changes in
  the pull request require changes to the reference documentation (either
  command-line reference, or API reference), those changes must be included as
  part of the pull request and will be reviewed now. Keep in mind that the
  narrative documentation may contain output examples of commands, so may need
  to be updated as well, in which case the `impact/documentation` label must
  be applied.
- If the PR has the `impact/documentation` label, merging is delayed until a
  documentation maintainer acknowledges that a corresponding documentation PR
  (or issue) is opened on the documentation repository. Once a documentation
  maintainer acknowledges the change, she/he will move the PR to `status/4-merge`
  for a code maintainer to push the green button.

Changes and additions to docs must be reviewed and approved (LGTM'd) by a minimum of two docs
sub-project maintainers. If the docs change originates with a docs maintainer, only one additional
LGTM is required (since we assume a docs maintainer approves of their own PR).

Once documentation is approved, a maintainer should make sure to remove this label and
add the next one.

Possible transitions from this state:

 * Close
 * `status/1-design-review`: new design concerns are raised
 * `status/2-code-review`: requires more code changes
 * `status/4-ready-to-merge`: general case

### Merge - `status/4-ready-to-merge`

Maintainers are expected to merge this pull request as soon as possible. They can ask for a rebase
or carry the pull request themselves.

Possible transitions from this state:

 * Merge: general case
 * Close: carry PR

After merging a pull request, the maintainer should consider applying one or multiple impact labels
to ease future classification:

 * `impact/api` signifies the patch impacted the Engine API
 * `impact/changelog` signifies the change is significant enough to make it in the changelog
 * `impact/cli` signifies the patch impacted a CLI command
 * `impact/dockerfile` signifies the patch impacted the Dockerfile syntax
 * `impact/deprecation` signifies the patch participates in deprecating an existing feature

### Close

If a pull request is closed it is expected that sufficient justification will be provided. In
particular, if there are alternative ways of achieving the same net result then those needs to be
spelled out. If the pull request is trying to solve a use case that is not one that we (as a
community) want to support then a justification for why should be provided.

The number of maintainers it takes to decide and close a PR is deliberately left unspecified. We
assume that the group of maintainers is bound by mutual trust and respect, and that opposition from
any single maintainer should be taken into consideration. Similarly, we expect maintainers to
justify their reasoning and to accept debating.

## Escalation process

Despite the previously described reviewing process, some PR might not show any progress for various
reasons:

 - No strong opinion for or against the proposed patch
 - Debates about the proper way to solve the problem at hand
 - Lack of consensus
 - ...

All these will eventually lead to stalled PR, where no apparent progress is made across several
weeks, or even months.

Maintainers should use their best judgement and apply the `status/needs-attention` label. It must
be used sparingly, as each PR with such label will be discussed by a group of maintainers during a
review session. The goal of that session is to agree on one of the following outcomes for the PR:

 * Close, explaining the rationale for not pursuing further
 * Continue, either by pushing the PR further in the workflow, or by deciding to carry the patch
   (ideally, a maintainer should be immediately assigned to make sure that the PR keeps continued
   attention)
 * Escalate to Solomon by formulating a few specific questions on which his answers will allow
   maintainers to decide.

## Milestones

Typically, every merged pull request get shipped naturally with the next release cut from the
`master` branch (either the next minor or major version, as indicated by the
[`VERSION`](https://github.com/docker/docker/blob/master/VERSION) file at the root of the
repository). However, the time-based nature of the release process provides no guarantee that a
given pull request will get merged in time. In other words, all open pull requests are implicitly
considered part of the next minor or major release milestone, and this won't be materialized on
GitHub.

A merged pull request must be attached to the milestone corresponding to the release in which it
will be shipped: this is both useful for tracking, and to help the release manager with the
changelog generation.

An open pull request may exceptionally get attached to a milestone to express a particular intent to
get it merged in time for that release. This may for example be the case for an important feature to
be included in a minor release, or a critical bugfix to be included in a patch release.

Finally, and as documented by the [`PATCH-RELEASES.md`](PATCH-RELEASES.md) process, the existence of
a milestone is not a guarantee that a release will happen, as some milestones will be created purely
for the purpose of bookkeeping
