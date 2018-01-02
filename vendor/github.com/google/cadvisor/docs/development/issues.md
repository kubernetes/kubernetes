# GitHub Issue tracking cAdvisor

This document outlines the process around GitHub issue tracking for cAdvisor at https://github.com/google/cadvisor/issues

## Labels

A brief explanation of what issue labels mean. Most labels also apply to pull requests, but for pull
requests which reference an issue, it is not necessary to copy the same labels to the PR.

- `area/API` - For issues related to the API.
- `area/UI` - For issues related to the web UI.
- `area/documentation` - For issues related to the documentation (inline comments or markdown).
- `area/performance` - For issues related to cAdvisor performance (speed, memory, etc.).
- `area/storage` - For issues related to cAdvisor storage plugins.
- `area/testing` - For issues related to testing (integration tests, unit tests, jenkins, etc.)
- `closed/duplicate` - For issues which have been closed as duplicates of another issue. The final
  comment on the issue should hold a reference the duplicate issue.
- `closed/infeasible` - For issues which cannot be resolved (e.g. a request for a feature we cannot
  or do not want to add).
- `community-assigned` - For issues which are being worked on by a community member (when github won't let us assign the issue to them).
- `kind/bug` - For issues referring to a bug in the existing implementation.
- `kind/enhancement` - For issues proposing an enhancement or new feature.
- `kind/support` - For issues which might just be user confusion / environment setup. If support
  issue ends up requiring a PR, it should probably be relabeled (for example, to `bug`). Many
  support issues may indicate a shortcoming of the documentation.
- `help wanted` - For issues which have been highlighted as a good place to contribute to
  cAdvisor. `help wanted` issues could be enhancements that the core team is unlikely to get to in
  the near future, or small projects which might be a good starting point. Lack of a `help wanted`
  label does not mean we won't accept contributions, it only means it was not identified as a
  candidate project for community contributions.
