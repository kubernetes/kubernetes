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

How to get your contributions merged smoothly and quickly:

- Create **small PRs** that are narrowly focused on **addressing a single
  concern**. We often receive PRs that attempt to fix several things at the same
  time, and if one part of the PR has a problem, that will hold up the entire
  PR.

- For **speculative changes**, consider opening an issue and discussing it
  first. If you are suggesting a behavioral or API change, consider starting
  with a [gRFC proposal](https://github.com/grpc/proposal). Many new features
  that are not bug fixes will require cross-language agreement.

- If you want to fix **formatting or style**, consider whether your changes are
  an obvious improvement or might be considered a personal preference. If a
  style change is based on preference, it likely will not be accepted. If it
  corrects widely agreed-upon anti-patterns, then please do create a PR and
  explain the benefits of the change.

- For correcting **misspellings**, please be aware that we use some terms that
  are sometimes flagged by spell checkers. As an example, "if an only if" is
  often written as "iff". Please do not make spelling correction changes unless
  you are certain they are misspellings.

- Provide a good **PR description** as a record of **what** change is being made
  and **why** it was made. Link to a GitHub issue if it exists.

- Maintain a **clean commit history** and use **meaningful commit messages**.
  PRs with messy commit histories are difficult to review and won't be merged.
  Before sending your PR, ensure your changes are based on top of the latest
  `upstream/master` commits, and avoid rebasing in the middle of a code review.
  You should **never use `git push -f`** unless absolutely necessary during a
  review, as it can interfere with GitHub's tracking of comments.

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

- If you are adding a new file, make sure it has the **copyright message**
  template at the top as a comment. You can copy the message from an existing
  file and update the year.

- The grpc package should only depend on standard Go packages and a small number
  of exceptions. **If your contribution introduces new dependencies**, you will
  need a discussion with gRPC-Go maintainers. A GitHub action check will run on
  every PR, and will flag any transitive dependency changes from any public
  package.

- Unless your PR is trivial, you should **expect reviewer comments** that you
  will need to address before merging. We'll label the PR as `Status: Requires
  Reporter Clarification` if we expect you to respond to these comments in a
  timely manner. If the PR remains inactive for 6 days, it will be marked as
  `stale`, and we will automatically close it after 7 days if we don't hear back
  from you. Please feel free to ping issues or bugs if you do not get a response
  within a week.

- Exceptions to the rules can be made if there's a compelling reason to do so.
