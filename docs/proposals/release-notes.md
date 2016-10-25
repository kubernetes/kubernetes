
# Kubernetes Release Notes

[djmm@google.com](mailto:djmm@google.com)<BR>
Last Updated: 2016-04-06

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Kubernetes Release Notes](#kubernetes-release-notes)
  - [Objective](#objective)
  - [Background](#background)
    - [The Problem](#the-problem)
    - [The (general) Solution](#the-general-solution)
      - [Then why not just list *every* change that was submitted, CHANGELOG-style?](#then-why-not-just-list-every-change-that-was-submitted-changelog-style)
  - [Options](#options)
  - [Collection Design](#collection-design)
  - [Publishing Design](#publishing-design)
    - [Location](#location)
    - [Layout](#layout)
      - [Alpha/Beta/Patch Releases](#alphabetapatch-releases)
      - [Major/Minor Releases](#majorminor-releases)
  - [Work estimates](#work-estimates)
  - [Caveats / Considerations](#caveats--considerations)

<!-- END MUNGE: GENERATED_TOC -->

## Objective

Define a process and design tooling for collecting, arranging and publishing
release notes for Kubernetes releases, automating as much of the process as
possible.

The goal is to introduce minor changes to the development workflow
in a way that is mostly frictionless and allows for the capture of release notes
as PRs are submitted to the repository.

This direct association of release notes to PRs captures the intention of
release visibility of the PR at the point an idea is submitted upstream.
The release notes can then be more easily collected and published when the
release is ready.

## Background

### The Problem

Release notes are often an afterthought and clarifying and finalizing them
is often left until the very last minute at the time the release is made.
This is usually long after the feature or bug fix was added and is no longer on
the mind of the author.  Worse, the collecting and summarizing of the
release is often left to those who may know little or nothing about these
individual changes!

Writing and editing release notes at the end of the cycle can be a rushed,
interrupt-driven and often stressful process resulting in incomplete,
inconsistent release notes often with errors and omissions.

### The (general) Solution

Like most things in the development/release pipeline, the earlier you do it,
the easier it is for everyone and the better the outcome.  Gather your release
notes earlier in the development cycle, at the time the features and fixes are
added.

#### Then why not just list *every* change that was submitted, CHANGELOG-style?

On larger projects like Kubernetes, showing every single change (PR) would mean
hundreds of entries.  The goal is to highlight the major changes for a release.

## Options

1. Use of pre-commit and other local git hooks
   * Experiments here using `prepare-commit-msg` and `commit-msg` git hook files
     were promising but less than optimal due to the fact that they would
     require input/confirmation with each commit and there may be multiple
     commits in a push and eventual PR.
1. Use of [github templates](https://github.com/blog/2111-issue-and-pull-request-templates)
   * Templates provide a great way to pre-fill PR comments, but there are no
     server-side hooks available to parse and/or easily check the contents of
     those templates to ensure that checkboxes were checked or forms were filled
     in.
1. Use of labels enforced by mungers/bots
   * We already make great use of mungers/bots to manage labels on PRs and it
     fits very nicely in the existing workflow

## Collection Design

The munger/bot option fits most cleanly into the existing workflow.

All `release-note-*` labeling is managed on the master branch PR only.
No `release-note-*` labels are needed on cherry-pick PRs and no information
will be collected from that cherry-pick PR.

The only exception to this rule is when a PR is not a cherry-pick and is
targeted directly to the non-master branch.  In this case, a `release-note-*`
label is required for that non-master PR.

1. New labels added to github: `release-note-none`, maybe others for new release note categories - see Layout section below
1. A [new munger](https://github.com/kubernetes/kubernetes/issues/23409) that will:
  * Add a `release-note-label-needed` label to all new master branch PRs
  * Block merge by the submit queue on all PRs labeled as `release-note-label-needed`
  * Auto-remove `release-note-label-needed` when one of the `release-note-*` labels is added

## Publishing Design

### Location

With v1.2.0, the release notes were moved from their previous [github releases](https://github.com/kubernetes/kubernetes/releases)
location to [CHANGELOG.md](../../CHANGELOG.md).  Going forward this seems like a good plan.
Other projects do similarly.

The kubernetes.tar.gz download link is also displayed along with the release notes
in [CHANGELOG.md](../../CHANGELOG.md).

Is there any reason to continue publishing anything to github releases if
the complete release story is published in [CHANGELOG.md](../../CHANGELOG.md)?

### Layout

Different types of releases will generally have different requirements in
terms of layout.  As expected, major releases like v1.2.0 are going
to require much more detail than the automated release notes will provide.

The idea is that these mechanisms will provide 100% of the release note
content for alpha, beta and most minor releases and bootstrap the content
with a release note 'template' for the authors of major releases like v1.2.0.

The authors can then collaborate and edit the higher level sections of the
release notes in a PR, updating [CHANGELOG.md](../../CHANGELOG.md) as needed.

v1.2.0 demonstrated the need, at least for major releases like v1.2.0, for
several sections in the published release notes.
In order to provide a basic layout for release notes in the future,
new releases can bootstrap [CHANGELOG.md](../../CHANGELOG.md) with the following template types:

#### Alpha/Beta/Patch Releases

These are automatically generated from `release-note*` labels, but can be modified as needed.

```
Action Required
* PR titles from the release-note-action-required label

Other notable changes
* PR titles from the release-note label
```

#### Major/Minor Releases

```
Major Themes
* Add to or delete this section

Other notable improvements
* Add to or delete this section

Experimental Features
* Add to or delete this section

Action Required
* PR titles from the release-note-action-required label

Known Issues
* Add to or delete this section

Provider-specific Notes
* Add to or delete this section

Other notable changes
* PR titles from the release-note label
```

## Work estimates

* The [new munger](https://github.com/kubernetes/kubernetes/issues/23409)
  * Owner: @eparis
  * Time estimate: Mostly done
* Updates to the tool that collects, organizes, publishes and sends release
  notifications.
  * Owner: @david-mcmahon
  * Time estimate: A few days


## Caveats / Considerations

* As part of the planning and development workflow how can we capture
  release notes for bigger features?
  [#23070](https://github.com/kubernetes/kubernetes/issues/23070)
  * For now contributors should simply use the first PR that enables a new
    feature by default.  We'll revisit if this does not work well.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/release-notes.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
