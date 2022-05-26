# Contributing Guidelines

source{d} go-git project is [Apache 2.0 licensed](LICENSE) and accepts
contributions via GitHub pull requests.  This document outlines some of the
conventions on development workflow, commit message formatting, contact points,
and other resources to make it easier to get your contribution accepted.

## Support Channels

The official support channels, for both users and contributors, are:

- [StackOverflow go-git tag](https://stackoverflow.com/questions/tagged/go-git) for user questions.
- GitHub [Issues](https://github.com/src-d/go-git/issues)* for bug reports and feature requests.

*Before opening a new issue or submitting a new pull request, it's helpful to
search the project - it's likely that another user has already reported the
issue you're facing, or it's a known issue that we're already aware of.


## How to Contribute

Pull Requests (PRs) are the main and exclusive way to contribute to the official go-git project.
In order for a PR to be accepted it needs to pass a list of requirements:

- You should be able to run the same query using `git`. We don't accept features that are not implemented in the official git implementation.
- The expected behavior must match the [official git implementation](https://github.com/git/git).
- The actual behavior must be correctly explained with natural language and providing a minimum working example in Go that reproduces it.
- All PRs must be written in idiomatic Go, formatted according to [gofmt](https://golang.org/cmd/gofmt/), and without any warnings from [go lint](https://github.com/golang/lint) nor [go vet](https://golang.org/cmd/vet/).
- They should in general include tests, and those shall pass.
- If the PR is a bug fix, it has to include a suite of unit tests for the new functionality.
- If the PR is a new feature, it has to come with a suite of unit tests, that tests the new functionality.
- In any case, all the PRs have to pass the personal evaluation of at least one of the maintainers of go-git.

### Format of the commit message

Every commit message should describe what was changed, under which context and, if applicable, the GitHub issue it relates to:

```
plumbing: packp, Skip argument validations for unknown capabilities. Fixes #623
```

The format can be described more formally as follows:

```
<package>: <subpackage>, <what changed>. [Fixes #<issue-number>]
```
