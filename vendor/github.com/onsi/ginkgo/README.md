![Ginkgo: A Go BDD Testing Framework](https://onsi.github.io/ginkgo/images/ginkgo.png)

[![Build Status](https://travis-ci.org/onsi/ginkgo.svg?branch=master)](https://travis-ci.org/onsi/ginkgo)

Jump to the [docs](https://onsi.github.io/ginkgo/) to learn more.  To start rolling your Ginkgo tests *now* [keep reading](#set-me-up)!

If you have a question, comment, bug report, feature request, etc. please open a GitHub issue, or visit the [Ginkgo Slack channel](https://app.slack.com/client/T029RQSE6/CQQ50BBNW).

## TLDR
Ginkgo builds on Go's `testing` package, allowing expressive [Behavior-Driven Development](https://en.wikipedia.org/wiki/Behavior-driven_development) ("BDD") style tests.
It is typically (and optionally) paired with the [Gomega](https://github.com/onsi/gomega) matcher library.

```go
Describe("the strings package", func() {
  Context("strings.Contains()", func() {
    When("the string contains the substring in the middle", func() {
      It("returns `true`", func() {
        Expect(strings.Contains("Ginkgo is awesome", "is")).To(BeTrue())
      })
    })
  })
})
```

## Feature List

- Ginkgo uses Go's `testing` package and can live alongside your existing `testing` tests.  It's easy to [bootstrap](https://onsi.github.io/ginkgo/#bootstrapping-a-suite) and start writing your [first tests](https://onsi.github.io/ginkgo/#adding-specs-to-a-suite)

- Ginkgo allows you to write tests in Go using expressive [Behavior-Driven Development](https://en.wikipedia.org/wiki/Behavior-driven_development) ("BDD") style:
    - Nestable [`Describe`, `Context` and `When` container blocks](https://onsi.github.io/ginkgo/#organizing-specs-with-containers-describe-and-context)
    - [`BeforeEach` and `AfterEach` blocks](https://onsi.github.io/ginkgo/#extracting-common-setup-beforeeach) for setup and teardown
    - [`It` and `Specify` blocks](https://onsi.github.io/ginkgo/#individual-specs-it) that hold your assertions
    - [`JustBeforeEach` blocks](https://onsi.github.io/ginkgo/#separating-creation-and-configuration-justbeforeeach) that separate creation from configuration (also known as the subject action pattern).
    - [`BeforeSuite` and `AfterSuite` blocks](https://onsi.github.io/ginkgo/#global-setup-and-teardown-beforesuite-and-aftersuite) to prep for and cleanup after a suite.

- A comprehensive test runner that lets you:
    - Mark specs as [pending](https://onsi.github.io/ginkgo/#pending-specs)
    - [Focus](https://onsi.github.io/ginkgo/#focused-specs) individual specs, and groups of specs, either programmatically or on the command line
    - Run your tests in [random order](https://onsi.github.io/ginkgo/#spec-permutation), and then reuse random seeds to replicate the same order.
    - Break up your test suite into parallel processes for straightforward [test parallelization](https://onsi.github.io/ginkgo/#parallel-specs)

- `ginkgo`: a command line interface with plenty of handy command line arguments for [running your tests](https://onsi.github.io/ginkgo/#running-tests) and [generating](https://onsi.github.io/ginkgo/#generators) test files.  Here are a few choice examples:
    - `ginkgo -nodes=N` runs your tests in `N` parallel processes and print out coherent output in realtime
    - `ginkgo -cover` runs your tests using Go's code coverage tool
    - `ginkgo convert` converts an XUnit-style `testing` package to a Ginkgo-style package
    - `ginkgo -focus="REGEXP"` and `ginkgo -skip="REGEXP"` allow you to specify a subset of tests to run via regular expression
    - `ginkgo -r` runs all tests suites under the current directory
    - `ginkgo -v` prints out identifying information for each tests just before it runs

    And much more: run `ginkgo help` for details!

    The `ginkgo` CLI is convenient, but purely optional -- Ginkgo works just fine with `go test`

- `ginkgo watch` [watches](https://onsi.github.io/ginkgo/#watching-for-changes) packages *and their dependencies* for changes, then reruns tests.  Run tests immediately as you develop!

- Built-in support for testing [asynchronicity](https://onsi.github.io/ginkgo/#asynchronous-tests)

- Built-in support for [benchmarking](https://onsi.github.io/ginkgo/#benchmark-tests) your code.  Control the number of benchmark samples as you gather runtimes and other, arbitrary, bits of numerical information about your code. 

- [Completions for Sublime Text](https://github.com/onsi/ginkgo-sublime-completions): just use [Package Control](https://sublime.wbond.net/) to install `Ginkgo Completions`.

- [Completions for VSCode](https://github.com/onsi/vscode-ginkgo): just use VSCode's extension installer to install `vscode-ginkgo`.

- Straightforward support for third-party testing libraries such as [Gomock](https://code.google.com/p/gomock/) and [Testify](https://github.com/stretchr/testify).  Check out the [docs](https://onsi.github.io/ginkgo/#third-party-integrations) for details.

- A modular architecture that lets you easily:
    - Write [custom reporters](https://onsi.github.io/ginkgo/#writing-custom-reporters) (for example, Ginkgo comes with a [JUnit XML reporter](https://onsi.github.io/ginkgo/#generating-junit-xml-output) and a TeamCity reporter).
    - [Adapt an existing matcher library (or write your own!)](https://onsi.github.io/ginkgo/#using-other-matcher-libraries) to work with Ginkgo

## [Gomega](https://github.com/onsi/gomega): Ginkgo's Preferred Matcher Library

Ginkgo is best paired with Gomega.  Learn more about Gomega [here](https://onsi.github.io/gomega/)

## [Agouti](https://github.com/sclevine/agouti): A Go Acceptance Testing Framework

Agouti allows you run WebDriver integration tests.  Learn more about Agouti [here](https://agouti.org)

## Getting Started

You'll need the Go command-line tools. Follow the [installation instructions](https://golang.org/doc/install) if you don't have it installed.

### Global installation
To install the Ginkgo command line interface into the `$PATH` (actually to `$GOBIN`):
```bash
go get -u github.com/onsi/ginkgo/ginkgo
```

### Go module ["tools package"](https://github.com/golang/go/issues/25922):
Create (or update) a file called `tools/tools.go` with the following contents:
```go
// +build tools

package tools

import (
	_ "github.com/onsi/ginkgo"
)

// This file imports packages that are used when running go generate, or used
// during the development process but not otherwise depended on by built code.
```
The Ginkgo command can then be run via `go run github.com/onsi/ginkgo/ginkgo`.
This approach allows the version of Ginkgo to be maintained under source control for reproducible results,
and is well suited to automated test pipelines.

### Bootstrapping
```bash
cd path/to/package/you/want/to/test

ginkgo bootstrap # set up a new ginkgo suite
ginkgo generate  # will create a sample test file.  edit this file and add your tests then...

go test # to run your tests

ginkgo  # also runs your tests

```

## I'm new to Go: What are my testing options?

Of course, I heartily recommend [Ginkgo](https://github.com/onsi/ginkgo) and [Gomega](https://github.com/onsi/gomega).  Both packages are seeing heavy, daily, production use on a number of projects and boast a mature and comprehensive feature-set.

With that said, it's great to know what your options are :)

### What Go gives you out of the box

Testing is a first class citizen in Go, however Go's built-in testing primitives are somewhat limited: The [testing](https://golang.org/pkg/testing) package provides basic XUnit style tests and no assertion library.

### Matcher libraries for Go's XUnit style tests

A number of matcher libraries have been written to augment Go's built-in XUnit style tests.  Here are two that have gained traction:

- [testify](https://github.com/stretchr/testify)
- [gocheck](https://labix.org/gocheck)

You can also use Ginkgo's matcher library [Gomega](https://github.com/onsi/gomega) in [XUnit style tests](https://onsi.github.io/gomega/#using-gomega-with-golangs-xunitstyle-tests)

### BDD style testing frameworks

There are a handful of BDD-style testing frameworks written for Go.  Here are a few:

- [Ginkgo](https://github.com/onsi/ginkgo) ;)
- [GoConvey](https://github.com/smartystreets/goconvey) 
- [Goblin](https://github.com/franela/goblin)
- [Mao](https://github.com/azer/mao)
- [Zen](https://github.com/pranavraja/zen)

Finally, @shageman has [put together](https://github.com/shageman/gotestit) a comprehensive comparison of Go testing libraries.

Go explore!

## License

Ginkgo is MIT-Licensed

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
