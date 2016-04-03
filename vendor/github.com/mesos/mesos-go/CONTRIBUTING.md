# Contributing
## Issues
When filing an issue, make sure to answer these five questions:

1. What version of the project are you using?
2. What operating system and processor architecture are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

## Code
### Proposals
Non trivial changes should be first discussed with the project maintainers by
opening a Github issue with the "Proposal: " title prefix, clearly explaining
rationale, context and implementation ideas.

This separate step is encouraged but not required.

### Implementation
Work should happen in an open pull request having a WIP prefix in its
title which gives visibility to the development process and provides
continuous integration feedback.

The pull request description must be well written and provide the necessary
context and background for review. If there's a proposal issue, it must be
referenced. When ready, replace the WIP prefix with PTAL which will
bring your contribution to the attention of project maintainers who will review
your PR in a timely manner.

Before review, keep in mind that:
- Git commit messages should conform to [community standards](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).
- Git commits should represent meaningful milestones or units of work.
- Changed or added code must be well tested. Different kinds of code
  require different testing strategies.
- Changed or added code must pass the project's CI.
- Changes to vendored files must be grouped into a single commit.

Once comments and revisions on the implementation wind down, the reviewers will
add the LGTM label which marks the PR as merge-able.
