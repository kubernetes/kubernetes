# Change LOG

**2018-11-16**
- added formatter output test suite, currently mainly pretty format
  tested.
- these tests, helped to identify some output format issues.

**2018-11-12**
- proper go module support added for `godog` command build.
- added build tests.

**2018-10-27**
- support go1.11 new compiler and linker changes for **godog** command.
- support go1.11 modules and `go mod` builds.
- `BindFlags` now has a prefix option for flags, so that `go test` command
  can avoid flag name collisions.
- `BindFlags` respect default options provided for binding, so that it
  does not override predefined options when flags are bind, see #144.
- Minor patch to support tag filters on example tables for
  ScenarioOutline.
- Minor patch for pretty printer, when scenario has no steps, comment
  possition computation was in panic.

**2018-03-04**
- support go1.10 new compiler and linker changes for **godog** command.

**2017-08-31**
- added **BeforeFeature** and **AfterFeature** hooks.
- failed multistep error is now prepended with a parent step text in order
  to determine failed nested step.
- pretty format now removes the step definition location package name in
  comment next to step if the step definition matches tested package. If
  step definition is imported from other package, full package name will
  be printed.

**2017-05-04**
- added **--strict** option in order to fail suite when there are pending
  or undefined steps. By default, suite passes and treats pending or
  undefined steps as TODOs.

**2017-04-29** - **v0.7.0**
- added support for nested steps. From now on, it is possible to return
  **godog.Steps** instead of an **error** in the step definition func.
  This change introduced few minor changes in **Formatter** interface. Be
  sure to adapt the changes if you have custom formatters.

**2017-04-27**
- added an option to randomize scenario execution order, so we could
  ensure that scenarios do not depend on global state.
- godog was manually sorting feature files by name. Now it just runs them
  in given order, you may sort them anyway you like. For example `godog
  $(find . -name '*.feature' | sort)`

**2016-10-30** - **v0.6.0**
- added experimental **events** format, this might be used for unified
  cucumber formats. But should be not adapted widely, since it is highly
  possible that specification will change.
- added **RunWithOptions** method which allows to easily run godog from
  **TestMain** without needing to simulate flag arguments. These options
  now allows to configure output writer.
- added flag **-o, --output=runner.binary** which only compiles the test
  runner executable, but does not execute it.
- **FlagSet** initialization now takes io.Writer as output for help text
  output. It was not showing nice colors on windows before.
  **--no-colors** option only applies to test run output.

**2016-06-14** - **v0.5.0**
- godog now uses **go tool compile** and **go tool link** to support
  vendor directory dependencies. It also compiles test executable the same
  way as standard **go test** utility. With this change, only go
  versions from **1.5** are now supported.

**2016-06-01**
- parse flags in main command, to show version and help without needing
  to compile test package and buildable go sources.

**2016-05-28**
- show nicely formatted called step func name and file path

**2016-05-26**
- pack gherkin dependency in a subpackage to prevent compatibility
  conflicts in the future. If recently upgraded, probably you will need to
  reference gherkin as `github.com/DATA-DOG/godog/gherkin` instead.

**2016-05-25**
- refactored test suite build tooling in order to use standard **go test**
  tool. Which allows to compile package with godog runner script in **go**
  idiomatic way. It also supports all build environment options as usual.
- **godog.Run** now returns an **int** exit status. It was not returning
  anything before, so there is no compatibility breaks.

**2016-03-04**
- added **junit** compatible output formatter, which prints **xml**
  results to **os.Stdout**
- fixed #14 which skipped printing background steps when there was
  scenario outline in feature.

**2015-07-03**
- changed **godog.Suite** from interface to struct. Context registration should be updated accordingly. The reason
for change: since it exports the same methods and there is no need to mock a function in tests, there is no
obvious reason to keep an interface.
- in order to support running suite concurrently, needed to refactor an entry point of application. The **Run** method
now is a func of godog package which initializes and run the suite (or more suites). Method **New** is removed. This
change made godog a little cleaner.
- renamed **RegisterFormatter** func to **Format** to be more consistent.

