# How to contribute

We welcome your patches and contributions to gRPC! Please read the gRPC
organization's [governance
rules](https://github.com/grpc/grpc-community/blob/master/governance.md) before
proceeding.

If you are new to GitHub, please start by reading [Pull Request howto](https://help.github.com/articles/about-pull-requests/)

## Legal requirements

In order to protect both you and ourselves, you will need to sign the
[Contributor License
Agreement](https://identity.linuxfoundation.org/projects/cncf). When you create
your first PR, a link will be added as a comment that contains the steps needed
to complete this process.

## Getting Started

A great way to start is by searching through our open issues. [Unassigned issues
labeled as "help
wanted"](https://github.com/grpc/grpc-go/issues?q=sort%3Aupdated-desc%20is%3Aissue%20is%3Aopen%20label%3A%22Status%3A%20Help%20Wanted%22%20no%3Aassignee)
are especially nice for first-time contributors, as they should be well-defined
problems that already have agreed-upon solutions.

## Code Style

We follow [Google's published Go style
guide](https://google.github.io/styleguide/go/). Note that there are three
primary documents that make up this style guide; please follow them as closely
as possible. If a reviewer recommends something that contradicts those
guidelines, there may be valid reasons to do so, but it should be rare.

## Guidelines for Pull Requests

Please read the following carefully to ensure your contributions can be merged
smoothly and quickly.

### PR Contents

- Create **small PRs** that are narrowly focused on **addressing a single
  concern**. We often receive PRs that attempt to fix several things at the same
  time, and if one part of the PR has a problem, that will hold up the entire
  PR.

- If your change does not address an **open issue** with an **agreed
  resolution**, consider opening an issue and discussing it first. If you are
  suggesting a behavioral or API change, consider starting with a [gRFC
  proposal](https://github.com/grpc/proposal). Many new features that are not
  bug fixes will require cross-language agreement.

- If you want to fix **formatting or style**, consider whether your changes are
  an obvious improvement or might be considered a personal preference. If a
  style change is based on preference, it likely will not be accepted. If it
  corrects widely agreed-upon anti-patterns, then please do create a PR and
  explain the benefits of the change.

- For correcting **misspellings**, please be aware that we use some terms that
  are sometimes flagged by spell checkers. As an example, "if an only if" is
  often written as "iff". Please do not make spelling correction changes unless
  you are certain they are misspellings.

- **All tests need to be passing** before your change can be merged. We
  recommend you run tests locally before creating your PR to catch breakages
  early on:

  - `./scripts/vet.sh` to catch vet errors.
  - `go test -cpu 1,4 -timeout 7m ./...` to run the tests.
  - `go test -race -cpu 1,4 -timeout 7m ./...` to run tests in race mode.

  Note that we have a multi-module repo, so `go test` commands may need to be
  run from the root of each module in order to cause all tests to run.

  *Alternatively*, you may find it easier to push your changes to your fork on
  GitHub, which will trigger a GitHub Actions run that you can use to verify
  everything is passing.

- Note that there are two GitHub actions checks that need not be green:

  1. We test the freshness of the generated proto code we maintain via the
     `vet-proto` check. If the source proto files are updated, but our repo is
     not updated, an optional checker will fail. This will be fixed by our team
     in a separate PR and will not prevent the merge of your PR.

  2. We run a checker that will fail if there is any change in dependencies of
     an exported package via the `dependencies` check. If new dependencies are
     added that are not appropriate, we may not accept your PR (see below).

- If you are adding a **new file**, make sure it has the **copyright message**
  template at the top as a comment. You can copy the message from an existing
  file and update the year.

- The grpc package should only depend on standard Go packages and a small number
  of exceptions. **If your contribution introduces new dependencies**, you will
  need a discussion with gRPC-Go maintainers.

### PR Descriptions

- **PR titles** should start with the name of the component being addressed, or
  the type of change. Examples: transport, client, server, round_robin, xds,
  cleanup, deps.

- Read and follow the **guidelines for PR titles and descriptions** here:
  https://google.github.io/eng-practices/review/developer/cl-descriptions.html

  *particularly* the sections "First Line" and "Body is Informative".

  Note: your PR description will be used as the git commit message in a
  squash-and-merge if your PR is approved. We may make changes to this as
  necessary.

- **Does this PR relate to an open issue?** On the first line, please use the
  tag `Fixes #<issue>` to ensure the issue is closed when the PR is merged. Or
  use `Updates #<issue>` if the PR is related to an open issue, but does not fix
  it. Consider filing an issue if one does not already exist.

- PR descriptions *must* conclude with **release notes** as follows:

  ```
  RELEASE NOTES:
  * <component>: <summary>
  ```

  This need not match the PR title.

  The summary must:

  * be something that gRPC users will understand.

  * clearly explain the feature being added, the issue being fixed, or the
    behavior being changed, etc. If fixing a bug, be clear about how the bug
    can be triggered by an end-user.

  * begin with a capital letter and use complete sentences.

  * be as short as possible to describe the change being made.

  If a PR is *not* end-user visible -- e.g. a cleanup, testing change, or
  GitHub-related, use `RELEASE NOTES: n/a`.

### PR Process

- Please **self-review** your code changes before sending your PR. This will
  prevent simple, obvious errors from causing delays.

- Maintain a **clean commit history** and use **meaningful commit messages**.
  PRs with messy commit histories are difficult to review and won't be merged.
  Before sending your PR, ensure your changes are based on top of the latest
  `upstream/master` commits, and avoid rebasing in the middle of a code review.
  You should **never use `git push -f`** unless absolutely necessary during a
  review, as it can interfere with GitHub's tracking of comments.

- Unless your PR is trivial, you should **expect reviewer comments** that you
  will need to address before merging. We'll label the PR as `Status: Requires
  Reporter Clarification` if we expect you to respond to these comments in a
  timely manner. If the PR remains inactive for 6 days, it will be marked as
  `stale`, and we will automatically close it after 7 days if we don't hear back
  from you. Please feel free to ping issues or bugs if you do not get a response
  within a week.
