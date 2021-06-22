# Contributing

We'd love your help making zap the very best structured logging library in Go!

If you'd like to add new exported APIs, please [open an issue][open-issue]
describing your proposal &mdash; discussing API changes ahead of time makes
pull request review much smoother. In your issue, pull request, and any other
communications, please remember to treat your fellow contributors with
respect! We take our [code of conduct](CODE_OF_CONDUCT.md) seriously.

Note that you'll need to sign [Uber's Contributor License Agreement][cla]
before we can accept any of your contributions. If necessary, a bot will remind
you to accept the CLA when you open your pull request.

## Setup

[Fork][fork], then clone the repository:

```
mkdir -p $GOPATH/src/go.uber.org
cd $GOPATH/src/go.uber.org
git clone git@github.com:your_github_username/zap.git
cd zap
git remote add upstream https://github.com/uber-go/zap.git
git fetch upstream
```

Make sure that the tests and the linters pass:

```
make test
make lint
```

If you're not using the minor version of Go specified in the Makefile's
`LINTABLE_MINOR_VERSIONS` variable, `make lint` doesn't do anything. This is
fine, but it means that you'll only discover lint failures after you open your
pull request.

## Making Changes

Start by creating a new branch for your changes:

```
cd $GOPATH/src/go.uber.org/zap
git checkout master
git fetch upstream
git rebase upstream/master
git checkout -b cool_new_feature
```

Make your changes, then ensure that `make lint` and `make test` still pass. If
you're satisfied with your changes, push them to your fork.

```
git push origin cool_new_feature
```

Then use the GitHub UI to open a pull request.

At this point, you're waiting on us to review your changes. We *try* to respond
to issues and pull requests within a few business days, and we may suggest some
improvements or alternatives. Once your changes are approved, one of the
project maintainers will merge them.

We're much more likely to approve your changes if you:

* Add tests for new functionality.
* Write a [good commit message][commit-message].
* Maintain backward compatibility.

[fork]: https://github.com/uber-go/zap/fork
[open-issue]: https://github.com/uber-go/zap/issues/new
[cla]: https://cla-assistant.io/uber-go/zap
[commit-message]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
