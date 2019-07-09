[![Build Status](https://travis-ci.org/DATA-DOG/godog.svg?branch=master)](https://travis-ci.org/DATA-DOG/godog)
[![GoDoc](https://godoc.org/github.com/DATA-DOG/godog?status.svg)](https://godoc.org/github.com/DATA-DOG/godog)
[![codecov.io](https://codecov.io/github/DATA-DOG/godog/branch/master/graph/badge.svg)](https://codecov.io/github/DATA-DOG/godog)

# Godog

<p align="center"><img src="/logo.png" alt="Godog logo" style="width:250px;" /></p>

**The API is likely to change a few times before we reach 1.0.0**

Please read all the README, you may find it very useful. And do not forget
to peek into the
[CHANGELOG](https://github.com/DATA-DOG/godog/blob/master/CHANGELOG.md)
from time to time.

Package godog is the official Cucumber BDD framework for Golang, it merges
specification and test documentation into one cohesive whole. The author
is a member of [cucumber team](https://github.com/cucumber).

The project is inspired by [behat][behat] and [cucumber][cucumber] and is
based on cucumber [gherkin3 parser][gherkin].

**Godog** does not intervene with the standard **go test** command
behavior. You can leverage both frameworks to functionally test your
application while maintaining all test related source code in **_test.go**
files.

**Godog** acts similar compared to **go test** command, by using go
compiler and linker tool in order to produce test executable. Godog
contexts need to be exported the same way as **Test** functions for go
tests. Note, that if you use **godog** command tool, it will use `go`
executable to determine compiler and linker.

**Godog** ships gherkin parser dependency as a subpackage. This will
ensure that it is always compatible with the installed version of godog.
So in general there are no vendor dependencies needed for installation.

The following about section was taken from
[cucumber](https://cucumber.io/) homepage.

## About

#### A single source of truth

Cucumber merges specification and test documentation into one cohesive whole.

#### Living documentation

Because they're automatically tested by Cucumber, your specifications are
always bang up-to-date.

#### Focus on the customer

Business and IT don't always understand each other. Cucumber's executable
specifications encourage closer collaboration, helping teams keep the
business goal in mind at all times.

#### Less rework

When automated testing is this much fun, teams can easily protect
themselves from costly regressions.

## Install

    go get github.com/DATA-DOG/godog/cmd/godog

## Example

The following example can be [found
here](/examples/godogs).

### Step 1

Given we create a new go package **$GOPATH/src/godogs**. From now on, this
is our work directory `cd $GOPATH/src/godogs`.

Imagine we have a **godog cart** to serve godogs for lunch. First of all,
we describe our feature in plain text - `vim
$GOPATH/src/godogs/features/godogs.feature`:

``` gherkin
# file: $GOPATH/src/godogs/features/godogs.feature
Feature: eat godogs
  In order to be happy
  As a hungry gopher
  I need to be able to eat godogs

  Scenario: Eat 5 out of 12
    Given there are 12 godogs
    When I eat 5
    Then there should be 7 remaining
```

**NOTE:** same as **go test** godog respects package level isolation. All
your step definitions should be in your tested package root directory. In
this case - `$GOPATH/src/godogs`

### Step 2

If godog is installed in your GOPATH. We can run `godog` inside the
**$GOPATH/src/godogs** directory. You should see that the steps are
undefined:

![Undefined step snippets](/screenshots/undefined.png?raw=true)

If we wish to vendor godog dependency, we can do it as usual, using tools
you prefer:

    git clone https://github.com/DATA-DOG/godog.git $GOPATH/src/godogs/vendor/github.com/DATA-DOG/godog

It gives you undefined step snippets to implement in your test context.
You may copy these snippets into your `godogs_test.go` file.

Our directory structure should now look like:

![Directory layout](/screenshots/dir-tree.png?raw=true)

If you copy the snippets into our test file and run godog again. We should
see the step definition is now pending:

![Pending step definition](/screenshots/pending.png?raw=true)

You may change **ErrPending** to **nil** and the scenario will
pass successfully.

Since we need a working implementation, we may start by implementing only what is necessary.

### Step 3

We only need a number of **godogs** for now. Lets keep it simple.

``` go
/* file: $GOPATH/src/godogs/godogs.go */
package main

// Godogs available to eat
var Godogs int

func main() { /* usual main func */ }
```

### Step 4

Now lets implement our step definitions, which we can copy from generated
console output snippets in order to test our feature requirements:

``` go
/* file: $GOPATH/src/godogs/godogs_test.go */
package main

import (
	"fmt"

	"github.com/DATA-DOG/godog"
)

func thereAreGodogs(available int) error {
	Godogs = available
	return nil
}

func iEat(num int) error {
	if Godogs < num {
		return fmt.Errorf("you cannot eat %d godogs, there are %d available", num, Godogs)
	}
	Godogs -= num
	return nil
}

func thereShouldBeRemaining(remaining int) error {
	if Godogs != remaining {
		return fmt.Errorf("expected %d godogs to be remaining, but there is %d", remaining, Godogs)
	}
	return nil
}

func FeatureContext(s *godog.Suite) {
	s.Step(`^there are (\d+) godogs$`, thereAreGodogs)
	s.Step(`^I eat (\d+)$`, iEat)
	s.Step(`^there should be (\d+) remaining$`, thereShouldBeRemaining)

	s.BeforeScenario(func(interface{}) {
		Godogs = 0 // clean the state before every scenario
	})
}
```

Now when you run the `godog` again, you should see:

![Passed suite](/screenshots/passed.png?raw=true)

We have hooked to **BeforeScenario** event in order to reset application
state before each scenario. You may hook into more events, like
**AfterStep** to print all state in case of an error. Or
**BeforeSuite** to prepare a database.

By now, you should have figured out, how to use **godog**. Another advice
is to make steps orthogonal, small and simple to read for an user. Whether
the user is a dumb website user or an API developer, who may understand
a little more technical context - it should target that user.

When steps are orthogonal and small, you can combine them just like you do
with Unix tools. Look how to simplify or remove ones, which can be
composed.

### References and Tutorials

- [cucumber-html-reporter](https://github.com/gkushang/cucumber-html-reporter)
  may be used in order to generate **html** reports together with
  **cucumber** output formatter. See the [following docker
  image](https://github.com/myie/cucumber-html-reporter) for usage
  details.
- [how to use godog by semaphoreci](https://semaphoreci.com/community/tutorials/how-to-use-godog-for-behavior-driven-development-in-go)
- see [examples](https://github.com/DATA-DOG/godog/tree/master/examples)
- see extension [AssistDog](https://github.com/hellomd/assistdog), which
  may have useful **gherkin.DataTable** transformations or comparison
  methods for assertions.

### Documentation

See [godoc][godoc] for general API details.
See **.travis.yml** for supported **go** versions.
See `godog -h` for general command options.

See implementation examples:

- [rest API server](/examples/api)
- [rest API with Database](/examples/db)
- [godogs](/examples/godogs)

## FAQ

### Running Godog with go test

You may integrate running **godog** in your **go test** command. You can
run it using go [TestMain](https://golang.org/pkg/testing/#hdr-Main) func
available since **go 1.4**. In this case it is not necessary to have
**godog** command installed. See the following examples.

The following example binds **godog** flags with specified prefix `godog`
in order to prevent flag collisions.

``` go
var opt = godog.Options{
	Output: colors.Colored(os.Stdout),
	Format: "progress", // can define default values
}

func init() {
	godog.BindFlags("godog.", flag.CommandLine, &opt)
}

func TestMain(m *testing.M) {
	flag.Parse()
	opt.Paths = flag.Args()

	status := godog.RunWithOptions("godogs", func(s *godog.Suite) {
		FeatureContext(s)
	}, opt)

	if st := m.Run(); st > status {
		status = st
	}
	os.Exit(status)
}
```

Then you may run tests with by specifying flags in order to filter
features.

```
go test -v --godog.random --godog.tags=wip
go test -v --godog.format=pretty --godog.random -race -coverprofile=coverage.txt -covermode=atomic
```

The following example does not bind godog flags, instead manually
configuring needed options.

``` go
func TestMain(m *testing.M) {
	status := godog.RunWithOptions("godog", func(s *godog.Suite) {
		FeatureContext(s)
	}, godog.Options{
		Format:    "progress",
		Paths:     []string{"features"},
		Randomize: time.Now().UTC().UnixNano(), // randomize scenario execution order
	})

	if st := m.Run(); st > status {
		status = st
	}
	os.Exit(status)
}
```

You can even go one step further and reuse **go test** flags, like
**verbose** mode in order to switch godog **format**. See the following
example:

``` go
func TestMain(m *testing.M) {
	format := "progress"
	for _, arg := range os.Args[1:] {
		if arg == "-test.v=true" { // go test transforms -v option
			format = "pretty"
			break
		}
	}
	status := godog.RunWithOptions("godog", func(s *godog.Suite) {
		godog.SuiteContext(s)
	}, godog.Options{
		Format: format,
		Paths:     []string{"features"},
	})

	if st := m.Run(); st > status {
		status = st
	}
	os.Exit(status)
}
```

Now when running `go test -v` it will use **pretty** format.

### Configure common options for godog CLI

There are no global options or configuration files. Alias your common or
project based commands: `alias godog-wip="godog --format=progress
--tags=@wip"`

### Testing browser interactions

**godog** does not come with builtin packages to connect to the browser.
You may want to look at [selenium](http://www.seleniumhq.org/) and
probably [phantomjs](http://phantomjs.org/). See also the following
components:

1. [browsersteps](https://github.com/llonchj/browsersteps) - provides
   basic context steps to start selenium and navigate browser content.
2. You may wish to have [goquery](https://github.com/PuerkitoBio/goquery)
   in order to work with HTML responses like with JQuery.

### Concurrency

In order to support concurrency well, you should reset the state and
isolate each scenario. They should not share any state. It is suggested to
run the suite concurrently in order to make sure there is no state
corruption or race conditions in the application.

It is also useful to randomize the order of scenario execution, which you
can now do with **--random** command option.

**NOTE:** if suite runs with concurrency option, it concurrently runs
every feature, not scenario per different features. This gives
a flexibility to isolate state per feature. For example using
**BeforeFeature** hook, it is possible to spin up costly service and shut
it down only in **AfterFeature** hook and share the service between all
scenarios in that feature. It is not advisable though, because you are
risking having a state dependency.

## Contributions

Feel free to open a pull request. Note, if you wish to contribute an extension to public (exported methods or types) -
please open an issue before to discuss whether these changes can be accepted. All backward incompatible changes are
and will be treated cautiously.

## License

**Godog** is licensed under the [three clause BSD license][license]

**Gherkin** is licensed under the [MIT][gherkin-license] and developed as
a part of the [cucumber project][cucumber]

[godoc]: http://godoc.org/github.com/DATA-DOG/godog "Documentation on godoc"
[golang]: https://golang.org/  "GO programming language"
[behat]: http://docs.behat.org/ "Behavior driven development framework for PHP"
[cucumber]: https://cucumber.io/ "Behavior driven development framework"
[gherkin]: https://github.com/cucumber/gherkin-go "Gherkin3 parser for GO"
[gherkin-license]: https://en.wikipedia.org/wiki/MIT_License "The MIT license"
[license]: http://en.wikipedia.org/wiki/BSD_licenses "The three clause BSD license"
