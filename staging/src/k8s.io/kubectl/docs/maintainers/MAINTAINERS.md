# SIG cli maintainers Guide

## Sustaining engineering tasks

The following tasks need to be performed consistently as a part of maintaining the health
of SIG cli.  We will be developing an oncall rotation for working on these tasks, where
the oncall is responsible to doing each task daily.

### Issue triage

Routinely monitor the newly filed issues and triage them to make sure we identify regressions.

[Kubectl repo](https://github.com/kubernetes/kubectl/issues)

[Kubernetes repo](https://github.com/kubernetes/kubernetes/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aopen%20label%3Asig%2Fcli)

Look for:

- Requests for help
  - Don't spend a lot of time on these, but answer and close them if it is easy  
- Regressions and bugs
  - Find the root cause
  - Triage the severity
  - Issues only occurring in old versions but not in new versions are less severe
- Simple issues for new contributors
  - Label these with "for-new-contributors"
  - Give them a priority
  - Make sure they are
    - Small
    - Well scoped
    - In areas of code with minimal technical debt
    - In areas of code with strong ownership already
- Feature requests
  - Do one of
    - Close them with an explanation along the lines of "Don't have capacity right now, try reopening in 6 months"
    - Label them with a "priority"

### Test triage

Monitor [test grid](https://k8s-testgrid.appspot.com/sig-cli-master)
and make sure the tests are passing.

If any tests are failing, debug them and send a fix.  Ask for help if you get stuck.

### PR review

Make sure PRs aren't getting stuck without attention.  If reviewers routinely don't respond
to PRs within a few days, we should take those reviewers out of the list.

Look through the PR list with [SIG cli](https://github.com/kubernetes/kubernetes/pulls?utf8=%E2%9C%93&q=is%3Apr%20is%3Aopen%20label%3Asig%2Fcli)

## New contributor assistance

- Look through issues labeled "for-new-contributors" that are assigned, and make sure they are active.
  If they haven't had activity in a couple days, ping the assignee and ask if help is needed.
- Identify issues for new contributors to pick up
- Figure out a progression for new contributors to become reviewers

## Per-release tasks

### At the start of the dev cycle

- Write planned features for each release
  - Use the [template](../template.md)

### During code-freeze

- Daily look at issues labeled with [sig/cli in the milestone](https://github.com/kubernetes/kubernetes/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aopen%20label%3Asig%2Fcli%20milestone%3Av1.9%20) and make sure they are owned and make progress
  - **Note:** You will need to update the milestone in the link to the current milestone

## Every 3-6 months tasks

### (3 months) Report about SIG cli at the community meeting

TODO: fill this in

### (6 months) Setup a SIG cli face-to-face

TODO: fill this in
