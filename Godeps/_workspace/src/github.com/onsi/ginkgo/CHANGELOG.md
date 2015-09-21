## HEAD

Improvements:

- `Skip(message)` can be used to skip the current test.

Bug Fixes:

- Ginkgo tests now fail when you `panic(nil)` (#167)

## 1.2.0 5/31/2015

Improvements

- `ginkgo -coverpkg` calls down to `go test -coverpkg` (#160)
- `ginkgo -afterSuiteHook COMMAND` invokes the passed-in `COMMAND` after a test suite completes (#152)
- Relaxed requirement for Go 1.4+.  `ginkgo` now works with Go v1.3+ (#166)

## 1.2.0-beta

Ginkgo now requires Go 1.4+

Improvements:

- Call reporters in reverse order when announcing spec completion -- allows custom reporters to emit output before the default reporter does.
- Improved focus behavior.  Now, this:

    ```golang
    FDescribe("Some describe", func() {
        It("A", func() {})

        FIt("B", func() {})
    })
    ```

  will run `B` but *not* `A`.  This tends to be a common usage pattern when in the thick of writing and debugging tests.
- When `SIGINT` is received, Ginkgo will emit the contents of the `GinkgoWriter` before running the `AfterSuite`.  Useful for debugging stuck tests.
- When `--progress` is set, Ginkgo will write test progress (in particular, Ginkgo will say when it is about to run a BeforeEach, AfterEach, It, etc...) to the `GinkgoWriter`.  This is useful for debugging stuck tests and tests that generate many logs.
- Improved output when an error occurs in a setup or teardown block.
- When `--dryRun` is set, Ginkgo will walk the spec tree and emit to its reporter *without* actually running anything.  Best paired with `-v` to understand which specs will run in which order.
- Add `By` to help document long `It`s.  `By` simply writes to the `GinkgoWriter`.
- Add support for precompiled tests:
    - `ginkgo build <path-to-package>` will now compile the package, producing a file named `package.test`
    - The compiled `package.test` file can be run directly.  This runs the tests in series.
    - To run precompiled tests in parallel, you can run: `ginkgo -p package.test`
- Support `bootstrap`ping and `generate`ing [Agouti](http://agouti.org) specs.
- `ginkgo generate` and `ginkgo bootstrap` now honor the package name already defined in a given directory
- The `ginkgo` CLI ignores `SIGQUIT`.  Prevents its stack dump from interlacing with the underlying test suite's stack dump.
- The `ginkgo` CLI now compiles tests into a temporary directory instead of the package directory.  This necessitates upgrading to Go v1.4+.
- `ginkgo -notify` now works on Linux

Bug Fixes:

- If --skipPackages is used and all packages are skipped, Ginkgo should exit 0.
- Fix tempfile leak when running in parallel
- Fix incorrect failure message when a panic occurs during a parallel test run
- Fixed an issue where a pending test within a focused context (or a focused test within a pending context) would skip all other tests.
- Be more consistent about handling SIGTERM as well as SIGINT
- When interupted while concurrently compiling test suites in the background, Ginkgo now cleans up the compiled artifacts.
- Fixed a long standing bug where `ginkgo -p` would hang if a process spawned by one of the Ginkgo parallel nodes does not exit. (Hooray!)

## 1.1.0 (8/2/2014)

No changes, just dropping the beta.

## 1.1.0-beta (7/22/2014)
New Features:

- `ginkgo watch` now monitors packages *and their dependencies* for changes.  The depth of the dependency tree can be modified with the `-depth` flag.
- Test suites with a programmatic focus (`FIt`, `FDescribe`, etc...) exit with non-zero status code, evne when they pass.  This allows CI systems to detect accidental commits of focused test suites.
- `ginkgo -p` runs the testsuite in parallel with an auto-detected number of nodes.
- `ginkgo -tags=TAG_LIST` passes a list of tags down to the `go build` command.
- `ginkgo --failFast` aborts the test suite after the first failure.
- `ginkgo generate file_1 file_2` can take multiple file arguments.
- Ginkgo now summarizes any spec failures that occured at the end of the test run. 
- `ginkgo --randomizeSuites` will run tests *suites* in random order using the generated/passed-in seed.

Improvements:

- `ginkgo -skipPackage` now takes a comma-separated list of strings.  If the *relative path* to a package matches one of the entries in the comma-separated list, that package is skipped.
- `ginkgo --untilItFails` no longer recompiles between attempts.
- Ginkgo now panics when a runnable node (`It`, `BeforeEach`, `JustBeforeEach`, `AfterEach`, `Measure`) is nested within another runnable node.  This is always a mistake.  Any test suites that panic because of this change should be fixed.

Bug Fixes:

- `ginkgo boostrap` and `ginkgo generate` no longer fail when dealing with `hyphen-separated-packages`.
- parallel specs are now better distributed across nodes - fixed a crashing bug where (for example) distributing 11 tests across 7 nodes would panic

## 1.0.0 (5/24/2014)
New Features:

- Add `GinkgoParallelNode()` - shorthand for `config.GinkgoConfig.ParallelNode`

Improvements:

- When compilation fails, the compilation output is rewritten to present a correct *relative* path.  Allows ⌘-clicking in iTerm open the file in your text editor.
- `--untilItFails` and `ginkgo watch` now generate new random seeds between test runs, unless a particular random seed is specified.

Bug Fixes:

- `-cover` now generates a correctly combined coverprofile when running with in parallel with multiple `-node`s.
- Print out the contents of the `GinkgoWriter` when `BeforeSuite` or `AfterSuite` fail.
- Fix all remaining race conditions in Ginkgo's test suite.

## 1.0.0-beta (4/14/2014)
Breaking changes:

- `thirdparty/gomocktestreporter` is gone.  Use `GinkgoT()` instead
- Modified the Reporter interface 
- `watch` is now a subcommand, not a flag.

DSL changes:

- `BeforeSuite` and `AfterSuite` for setting up and tearing down test suites.
- `AfterSuite` is triggered on interrupt (`^C`) as well as exit.
- `SynchronizedBeforeSuite` and `SynchronizedAfterSuite` for setting up and tearing down singleton resources across parallel nodes.

CLI changes:

- `watch` is now a subcommand, not a flag
- `--nodot` flag can be passed to `ginkgo generate` and `ginkgo bootstrap` to avoid dot imports.  This explicitly imports all exported identifiers in Ginkgo and Gomega.  Refreshing this list can be done by running `ginkgo nodot`
- Additional arguments can be passed to specs.  Pass them after the `--` separator
- `--skipPackage` flag takes a regexp and ignores any packages with package names passing said regexp.
- `--trace` flag prints out full stack traces when errors occur, not just the line at which the error occurs.

Misc:

- Start using semantic versioning
- Start maintaining changelog

Major refactor:

- Pull out Ginkgo's internal to `internal`
- Rename `example` everywhere to `spec`
- Much more!
