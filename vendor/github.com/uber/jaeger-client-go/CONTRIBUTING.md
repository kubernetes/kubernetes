# Contributing to `jaeger-client-go`

We'd love your help! If you would like to contribute code you can do so through GitHub
by forking the repository and sending a pull request into the `master` branch.

## Getting Started

This library uses [glide](https://github.com/Masterminds/glide) to manage dependencies.

To get started:

```bash
git submodule update --init --recursive
glide install
make test
```

## Making A Change

*Before making any significant changes, please [open an
issue](https://github.com/uber/jaeger-client-go/issues).* Discussing your proposed
changes ahead of time will make the contribution process smooth for everyone.

Once we've discussed your changes and you've got your code ready, make sure
that tests are passing (`make test` or `make cover`) and open your PR! Your
pull request is most likely to be accepted if it:

* Includes tests for new functionality.
* Follows the guidelines in [Effective
  Go](https://golang.org/doc/effective_go.html) and the [Go team's common code
  review comments](https://github.com/golang/go/wiki/CodeReviewComments).
* Has a [good commit
  message](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).

## Cutting a Release

1. Create a PR "Preparing for release X.Y.Z" against master branch
    * Alter CHANGELOG.md from `<placeholder_version> (unreleased)` to `<X.Y.Z> (YYYY-MM-DD)`
    * Update `JaegerClientVersion` in constants.go to `Go-X.Y.Z`
2. Create a release "Release X.Y.Z" on Github
    * Create Tag `vX.Y.Z`
    * Copy CHANGELOG.md into the release notes
3. Create a PR "Back to development" against master branch
    * Add `<next_version> (unreleased)` to CHANGELOG.md
    * Update `JaegerClientVersion` in constants.go to `Go-<next_version>dev`

## License

By contributing your code, you agree to license your contribution under the terms
of the MIT License: https://github.com/uber/jaeger-client-go/blob/master/LICENSE

If you are adding a new file it should have a header like below.

```
// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
```

