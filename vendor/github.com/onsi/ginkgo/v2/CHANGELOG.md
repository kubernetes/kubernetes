## 2.19.0

### Features

[Label Sets](https://onsi.github.io/ginkgo/#label-sets) allow for more expressive and flexible label filtering.

## 2.18.0

### Features
- Add --slience-skips and --force-newlines [f010b65]
- fail when no tests were run and --fail-on-empty was set [d80eebe]

### Fixes
- Fix table entry context edge case [42013d6]

### Maintenance
- Bump golang.org/x/tools from 0.20.0 to 0.21.0 (#1406) [fcf1fd7]
- Bump github.com/onsi/gomega from 1.33.0 to 1.33.1 (#1399) [8bb14fd]
- Bump golang.org/x/net from 0.24.0 to 0.25.0 (#1407) [04bfad7]

## 2.17.3

### Fixes
`ginkgo watch` now ignores hidden files [bde6e00]

## 2.17.2

### Fixes
- fix: close files [32259c8]
- fix github output log level for skipped specs [780e7a3]

### Maintenance
- Bump github.com/google/pprof [d91fe4e]
- Bump github.com/go-task/slim-sprig to v3 [8cb662e]
- Bump golang.org/x/net in /integration/_fixtures/version_mismatch_fixture (#1391) [3134422]
- Bump github-pages from 230 to 231 in /docs (#1384) [eca81b4]
- Bump golang.org/x/tools from 0.19.0 to 0.20.0 (#1383) [760def8]
- Bump golang.org/x/net from 0.23.0 to 0.24.0 (#1381) [4ce33f4]
- Fix test for gomega version bump [f2fcd97]
- Bump github.com/onsi/gomega from 1.30.0 to 1.33.0 (#1390) [fd622d2]
- Bump golang.org/x/tools from 0.17.0 to 0.19.0 (#1368) [5474a26]
- Bump github-pages from 229 to 230 in /docs (#1359) [e6d1170]
- Bump google.golang.org/protobuf from 1.28.0 to 1.33.0 (#1374) [7f447b2]
- Bump golang.org/x/net from 0.20.0 to 0.23.0 (#1380) [f15239a]

## 2.17.1

### Fixes
- If the user sets --seed=0, make sure all parallel nodes get the same seed [af0330d]

## 2.17.0

### Features

- add `--github-output` for nicer output in github actions [e8a2056]

### Maintenance

- fix typo in core_dsl.go [977bc6f]
- Fix typo in docs [e297e7b]

## 2.16.0

### Features
- add SpecContext to reporting nodes

### Fixes
- merge coverages instead of combining them (#1329) (#1340) [23f0cc5]
- core_dsl: disable Getwd() with environment variable (#1357) [cd418b7]

### Maintenance
- docs/index.md: Typo [2cebe8d]
- fix docs [06de431]
- chore: test with Go 1.22 (#1352) [898cba9]
- Bump golang.org/x/tools from 0.16.1 to 0.17.0 (#1336) [17ae120]
- Bump golang.org/x/sys from 0.15.0 to 0.16.0 (#1327) [5a179ed]
- Bump github.com/go-logr/logr from 1.3.0 to 1.4.1 (#1321) [a1e6b69]
- Bump github-pages and jekyll-feed in /docs (#1351) [d52951d]
- Fix docs for handling failures in goroutines (#1339) [4471b2e]

## 2.15.0

### Features

- JUnit reports now interpret Label(owner:X) and set owner to X. [8f3bd70]
- include cancellation reason when cancelling spec context [96e915c]

### Fixes

- emit output of failed go tool cover invocation so users can try to debug things for themselves [c245d09]
- fix outline when using nodot in ginkgo v2 [dca77c8]
- Document areas where GinkgoT() behaves differently from testing.T [dbaf18f]
- bugfix(docs): use Unsetenv instead of Clearenv (#1337) [6f67a14]

### Maintenance

- Bump to go 1.20 [4fcd0b3]

## 2.14.0

### Features
You can now use `GinkgoTB()` when you need an instance of `testing.TB` to pass to a library.

Prior to this release table testing only supported generating individual `It`s for each test entry.  `DescribeTableSubtree` extends table testing support to entire testing subtrees - under the hood `DescrieTableSubtree` generates a new container for each entry and invokes your function to fill our the container.  See the [docs](https://onsi.github.io/ginkgo/#generating-subtree-tables) to learn more.

- Introduce DescribeTableSubtree [65ec56d]
- add GinkgoTB() to docs [4a2c832]
- Add GinkgoTB() function (#1333) [92b6744]

### Fixes
- Fix typo in internal/suite.go (#1332) [beb9507]
- Fix typo in docs/index.md (#1319) [4ac3a13]
- allow wasm to compile with ginkgo present (#1311) [b2e5bc5]

### Maintenance
- Bump golang.org/x/tools from 0.16.0 to 0.16.1 (#1316) [465a8ec]
- Bump actions/setup-go from 4 to 5 (#1313) [eab0e40]
- Bump github/codeql-action from 2 to 3 (#1317) [fbf9724]
- Bump golang.org/x/crypto (#1318) [3ee80ee]
- Bump golang.org/x/tools from 0.14.0 to 0.16.0 (#1306) [123e1d5]
- Bump github.com/onsi/gomega from 1.29.0 to 1.30.0 (#1297) [558f6e0]
- Bump golang.org/x/net from 0.17.0 to 0.19.0 (#1307) [84ff7f3]

## 2.13.2

### Fixes
- Fix file handler leak (#1309) [e2e81c8]
- Avoid allocations with `(*regexp.Regexp).MatchString` (#1302) [3b2a2a7]

## 2.13.1

### Fixes
- # 1296 fix(precompiled test guite): exec bit check omitted on Windows (#1301) [26eea01]

### Maintenance
- Bump github.com/go-logr/logr from 1.2.4 to 1.3.0 (#1291) [7161a9d]
- Bump golang.org/x/sys from 0.13.0 to 0.14.0 (#1295) [7fc7b10]
- Bump golang.org/x/tools from 0.12.0 to 0.14.0 (#1282) [74bbd65]
- Bump github.com/onsi/gomega from 1.27.10 to 1.29.0 (#1290) [9373633]
- Bump golang.org/x/net in /integration/_fixtures/version_mismatch_fixture (#1286) [6e3cf65]

## 2.13.0

### Features

Add PreviewSpect() to enable programmatic preview access to the suite report (fixes #1225)

## 2.12.1

### Fixes
- Print logr prefix if it exists (#1275) [90d4846]

### Maintenance
- Bump actions/checkout from 3 to 4 (#1271) [555f543]
- Bump golang.org/x/sys from 0.11.0 to 0.12.0 (#1270) [d867b7d]

## 2.12.0

### Features

- feat: allow MustPassRepeatedly decorator to be set at suite level (#1266) [05de518]

### Fixes

- fix-errors-in-readme (#1244) [27c2f5d]

### Maintenance

Various chores/dependency bumps.

## 2.11.0

In prior versions of Ginkgo specs the CLI filter flags (e.g. `--focus`, `--label-filter`) would _override_ any programmatic focus.  This behavior has proved surprising and confusing in at least the following ways:

- users cannot combine programmatic filters and CLI filters to more efficiently select subsets of tests
- CLI filters can override programmatic focus on CI systems resulting in an exit code of 0 despite the presence of (incorrectly!) committed focused specs.

Going forward Ginkgo will AND all programmatic and CLI filters.  Moreover, the presence of any programmatic focused tests will always result in a non-zero exit code.

This change is technically a change in Ginkgo's external contract and may require some users to make changes to successfully adopt. Specifically: it's possible some users were intentionally using CLI filters to override programmatic focus.  If this is you please open an issue so we can explore solutions to the underlying problem you are trying to solve.

### Fixes
- Programmatic focus is no longer overwrriten by CLI filters [d6bba86]

### Maintenance
- Bump github.com/onsi/gomega from 1.27.7 to 1.27.8 (#1218) [4a70a38]
- Bump golang.org/x/sys from 0.8.0 to 0.9.0 (#1219) [97eda4d]

## 2.10.0

### Features
- feat(ginkgo/generators): add --tags flag (#1216) [a782a77]
  adds a new --tags flag to ginkgo generate

### Fixes
- Fix broken link of MIGRATING_TO_V2.md (#1217) [548d78e]

### Maintenance
- Bump golang.org/x/tools from 0.9.1 to 0.9.3 (#1215) [2b76a5e]

## 2.9.7

### Fixes
- fix race when multiple defercleanups are called in goroutines [07fc3a0]

## 2.9.6

### Fixes
- fix: create parent directory before report files (#1212) [0ac65de]

### Maintenance
- Bump github.com/onsi/gomega from 1.27.6 to 1.27.7 (#1202) [3e39231]

## 2.9.5

### Fixes
- ensure the correct deterministic sort order is produced when ordered specs are generated by a helper function [7fa0b6b]

### Maintenance
- fix generators link (#1200) [9f9d8b9]
- Bump golang.org/x/tools from 0.8.0 to 0.9.1 (#1196) [150e3f2]
- fix spelling err in docs (#1199) [0013b1a]
- Bump golang.org/x/sys from 0.7.0 to 0.8.0 (#1193) [9e9e3e5]

## 2.9.4

### Fixes
- fix hang with ginkgo -p (#1192) [15d4bdc] - this addresses a _long_ standing issue related to Ginkgo hanging when a child process spawned by the test does not exit.

- fix: fail fast may cause Serial spec or cleanup Node interrupted (#1178) [8dea88b] - prior to this there was a small gap in which specs on other processes might start even if one process has tried to abort the suite.


### Maintenance
- Document run order when multiple setup nodes are at the same nesting level [903be81]

## 2.9.3

### Features
- Add RenderTimeline to GinkgoT() [c0c77b6]

### Fixes
- update Measure deprecation message. fixes #1176 [227c662]
- add newlines to GinkgoLogr (#1170) (#1171) [0de0e7c]

### Maintenance
- Bump commonmarker from 0.23.8 to 0.23.9 in /docs (#1183) [8b925ab]
- Bump nokogiri from 1.14.1 to 1.14.3 in /docs (#1184) [e3795a4]
- Bump golang.org/x/tools from 0.7.0 to 0.8.0 (#1182) [b453793]
- Bump actions/setup-go from 3 to 4 (#1164) [73ed75b]
- Bump github.com/onsi/gomega from 1.27.4 to 1.27.6 (#1173) [0a2bc64]
- Bump github.com/go-logr/logr from 1.2.3 to 1.2.4 (#1174) [f41c557]
- Bump golang.org/x/sys from 0.6.0 to 0.7.0 (#1179) [8e423e5]

## 2.9.2

### Maintenance
- Bump github.com/go-task/slim-sprig (#1167) [3fcc5bf]
- Bump github.com/onsi/gomega from 1.27.3 to 1.27.4 (#1163) [6143ffe]

## 2.9.1

### Fixes
This release fixes a longstanding issue where `ginkgo -coverpkg=./...` would not work.  This is now resolved and fixes [#1161](https://github.com/onsi/ginkgo/issues/1161) and [#995](https://github.com/onsi/ginkgo/issues/995)
- Support -coverpkg=./... [26ca1b5]
- document coverpkg a bit more clearly [fc44c3b]

### Maintenance
- bump various dependencies
- Improve Documentation and fix typo (#1158) [93de676]

## 2.9.0

### Features
- AttachProgressReporter is an experimental feature that allows users to provide arbitrary information when a ProgressReport is requested [28801fe]

- GinkgoT() has been expanded to include several Ginkgo-specific methods [2bd5a3b]

  The intent is to enable the development of third-party libraries that integrate deeply with Ginkgo using `GinkgoT()` to access Ginkgo's functionality.

## 2.8.4

### Features
- Add OmitSuiteSetupNodes to JunitReportConfig (#1147) [979fbc2]
- Add a reference to ginkgolinter in docs.index.md (#1143) [8432589]

### Fixes
- rename tools hack to see if it fixes things for downstream users [a8bb39a]

### Maintenance
- Bump golang.org/x/text (#1144) [41b2a8a]
- Bump github.com/onsi/gomega from 1.27.0 to 1.27.1 (#1142) [7c4f583]

## 2.8.3

Released to fix security issue in golang.org/x/net dependency

### Maintenance

- Bump golang.org/x/net from 0.6.0 to 0.7.0 (#1141) [fc1a02e]
- remove tools.go hack from documentation [0718693]

## 2.8.2

Ginkgo now includes a `tools.go` file in the root directory of the `ginkgo` package.  This should allow modules that simply `go get github.com/onsi/ginkgo/v2` to also pull in the CLI dependencies.  This obviates the need for consumers of Ginkgo to have their own `tools.go` file and makes it simpler to ensure that the version of the `ginkgo` CLI being used matches the version of the library.  You can simply run `go run github.com/onsi/ginkgo/v2/ginkgo` to run the version of the cli associated with your package go.mod.

### Maintenance

- Bump github.com/onsi/gomega from 1.26.0 to 1.27.0 (#1139) [5767b0a]
- Fix minor typos (#1138) [e1e9723]
- Fix link in V2 Migration Guide (#1137) [a588f60]

## 2.8.1

### Fixes
- lock around default report output to avoid triggering the race detector when calling By from goroutines [2d5075a]
- don't run ReportEntries through sprintf [febbe38]

### Maintenance
- Bump golang.org/x/tools from 0.5.0 to 0.6.0 (#1135) [11a4860]
- test: update matrix for Go 1.20 (#1130) [4890a62]
- Bump golang.org/x/sys from 0.4.0 to 0.5.0 (#1133) [a774638]
- Bump github.com/onsi/gomega from 1.25.0 to 1.26.0 (#1120) [3f233bd]
- Bump github-pages from 227 to 228 in /docs (#1131) [f9b8649]
- Bump activesupport from 6.0.6 to 6.0.6.1 in /docs (#1127) [6f8c042]
- Update index.md with instructions on how to upgrade Ginkgo [833a75e]

## 2.8.0

### Features

- Introduce GinkgoHelper() to track and exclude helper functions from potential CodeLocations [e19f556]

Modeled after `testing.T.Helper()`.  Now, rather than write code like:

```go
func helper(model Model) {
    Expect(model).WithOffset(1).To(BeValid())
    Expect(model.SerialNumber).WithOffset(1).To(MatchRegexp(/[a-f0-9]*/))
}
```

you can stop tracking offsets (which makes nesting composing helpers nearly impossible) and simply write:

```go
func helper(model Model) {
    GinkgoHelper()
    Expect(model).To(BeValid())
    Expect(model.SerialNumber).To(MatchRegexp(/[a-f0-9]*/))
}
```

- Introduce GinkgoLabelFilter() and Label().MatchesLabelFilter() to make it possible to programmatically match filters (fixes #1119) [2f6597c]

You can now write code like this:

```go
BeforeSuite(func() {
	if Label("slow").MatchesLabelFilter(GinkgoLabelFilter()) {
		// do slow setup
	}

	if Label("fast").MatchesLabelFilter(GinkgoLabelFilter()) {
		// do fast setup
	}
})
```

to programmatically check whether a given set of labels will match the configured `--label-filter`.

### Maintenance

- Bump webrick from 1.7.0 to 1.8.1 in /docs (#1125) [ea4966e]
- cdeql: add ruby language (#1124) [9dd275b]
- dependabot: add bundler package-ecosystem for docs (#1123) [14e7bdd]

## 2.7.1

### Fixes
- Bring back SuiteConfig.EmitSpecProgress to avoid compilation issue for consumers that set it manually [d2a1cb0]

### Maintenance
- Bump github.com/onsi/gomega from 1.24.2 to 1.25.0 (#1118) [cafece6]
- Bump golang.org/x/tools from 0.4.0 to 0.5.0 (#1111) [eda66c2]
- Bump golang.org/x/sys from 0.3.0 to 0.4.0 (#1112) [ac5ccaa]
- Bump github.com/onsi/gomega from 1.24.1 to 1.24.2 (#1097) [eee6480]

## 2.7.0

### Features
- Introduce ContinueOnFailure for Ordered containers [e0123ca] - Ordered containers that are also decorated with ContinueOnFailure will not stop running specs after the first spec fails.
- Support for bootstrap commands to use custom data for templates (#1110) [7a2b242]
- Support for labels and pending decorator in ginkgo outline output (#1113) [e6e3b98]
- Color aliases for custom color support (#1101) [49fab7a]

### Fixes
- correctly ensure deterministic spec order, even if specs are generated by iterating over a map [89dda20]
- Fix a bug where timedout specs were not correctly treated as failures when determining whether or not to run AfterAlls in an Ordered container.
- Ensure go test coverprofile outputs to the expected location (#1105) [b0bd77b]

## 2.6.1

### Features
- Override formatter colors from envvars - this is a new feature but an alternative approach involving config files might be taken in the future (#1095) [60240d1]

### Fixes
- GinkgoRecover now supports ignoring panics that match a specific, hidden, interface [301f3e2]

### Maintenance
- Bump github.com/onsi/gomega from 1.24.0 to 1.24.1 (#1077) [3643823]
- Bump golang.org/x/tools from 0.2.0 to 0.4.0 (#1090) [f9f856e]
- Bump nokogiri from 1.13.9 to 1.13.10 in /docs (#1091) [0d7087e]

## 2.6.0

### Features
- `ReportBeforeSuite` provides access to the suite report before the suite begins.
- Add junit config option for omitting leafnodetype (#1088) [956e6d2]
- Add support to customize junit report config to omit spec labels (#1087) [de44005]

### Fixes
- Fix stack trace pruning so that it has a chance of working on windows [2165648]

## 2.5.1

### Fixes
- skipped tests only show as 'S' when running with -v [3ab38ae]
- Fix typo in docs/index.md (#1082) [55fc58d]
- Fix typo in docs/index.md (#1081) [8a14f1f]
- Fix link notation in docs/index.md (#1080) [2669612]
- Fix typo in `--progress` deprecation message (#1076) [b4b7edc]

### Maintenance
- chore: Included githubactions in the dependabot config (#976) [baea341]
- Bump golang.org/x/sys from 0.1.0 to 0.2.0 (#1075) [9646297]

## 2.5.0

### Ginkgo output now includes a timeline-view of the spec

This commit changes Ginkgo's default output.  Spec details are now
presented as a **timeline** that includes events that occur during the spec
lifecycle interleaved with any GinkgoWriter content.  This makes is much easier
to understand the flow of a spec and where a given failure occurs.

The --progress, --slow-spec-threshold, --always-emit-ginkgo-writer flags
and the SuppressProgressReporting decorator have all been deprecated.  Instead
the existing -v and -vv flags better capture the level of verbosity to display.  However,
a new --show-node-events flag is added to include node `> Enter` and `< Exit` events
in the spec timeline.

In addition, JUnit reports now include the timeline (rendered with -vv) and custom JUnit
reports can be configured and generated using
`GenerateJUnitReportWithConfig(report types.Report, dst string, config JunitReportConfig)`

Code should continue to work unchanged with this version of Ginkgo - however if you have tooling that
was relying on the specific output format of Ginkgo you _may_ run into issues.  Ginkgo's console output is not guaranteed to be stable for tooling and automation purposes.  You should, instead, use Ginkgo's JSON format
to build tooling on top of as it has stronger guarantees to be stable from version to version.

### Features
- Provide details about which timeout expired [0f2fa27]

### Fixes
- Add Support Policy to docs [c70867a]

### Maintenance
- Bump github.com/onsi/gomega from 1.22.1 to 1.23.0 (#1070) [bb3b4e2]

## 2.4.0

### Features

- DeferCleanup supports functions with multiple-return values [5e33c75]
- Add GinkgoLogr (#1067) [bf78c28]
- Introduction of 'MustPassRepeatedly' decorator (#1051) [047c02f]

### Fixes
- correcting some typos (#1064) [1403d3c]
- fix flaky internal_integration interrupt specs [2105ba3]
- Correct busted link in README [be6b5b9]

### Maintenance
- Bump actions/checkout from 2 to 3 (#1062) [8a2f483]
- Bump golang.org/x/tools from 0.1.12 to 0.2.0 (#1065) [529c4e8]
- Bump github/codeql-action from 1 to 2 (#1061) [da09146]
- Bump actions/setup-go from 2 to 3 (#1060) [918040d]
- Bump github.com/onsi/gomega from 1.22.0 to 1.22.1 (#1053) [2098e4d]
- Bump nokogiri from 1.13.8 to 1.13.9 in /docs (#1066) [1d74122]
- Add GHA to dependabot config [4442772]

## 2.3.1

## Fixes
Several users were invoking `ginkgo` by installing the latest version of the cli via `go install github.com/onsi/ginkgo/v2/ginkgo@latest`.  When 2.3.0 was released this resulted in an influx of issues as CI systems failed due to a change in the internal contract between the Ginkgo CLI and the Ginkgo library.  Ginkgo only supports running the same version of the library as the cli (which is why both are packaged in the same repository).

With this patch release, the ginkgo CLI can now identify a version mismatch and emit a helpful error message.

- Ginkgo cli can identify version mismatches and emit a helpful error message [bc4ae2f]
- further emphasize that a version match is required when running Ginkgo on CI and/or locally [2691dd8]

### Maintenance
- bump gomega to v1.22.0 [822a937]

## 2.3.0

### Interruptible Nodes and Timeouts

Ginkgo now supports per-node and per-spec timeouts on interruptible nodes.  Check out the [documentation for all the details](https://onsi.github.io/ginkgo/#spec-timeouts-and-interruptible-nodes) but the gist is you can now write specs like this:

```go
It("is interruptible", func(ctx SpecContext) { // or context.Context instead of SpecContext, both are valid.
    // do things until `ctx.Done()` is closed, for example:
    req, err := http.NewRequestWithContext(ctx, "POST", "/build-widgets", nil)
    Expect(err).NotTo(HaveOccured())
    _, err := http.DefaultClient.Do(req)
    Expect(err).NotTo(HaveOccured())

    Eventually(client.WidgetCount).WithContext(ctx).Should(Equal(17))
}, NodeTimeout(time.Second*20), GracePeriod(5*time.Second))
```

and have Ginkgo ensure that the node completes before the timeout elapses.  If it does elapse, or if an external interrupt is received (e.g. `^C`) then Ginkgo will cancel the context and wait for the Grace Period for the node to exit before proceeding with any cleanup nodes associated with the spec.  The `ctx` provided by Ginkgo can also be passed down to Gomega's `Eventually` to have all assertions within the node governed by a single deadline.

### Features

- Ginkgo now records any additional failures that occur during the cleanup of a failed spec.  In prior versions this information was quietly discarded, but the introduction of a more rigorous approach to timeouts and interruptions allows Ginkgo to better track subsequent failures.
- `SpecContext` also provides a mechanism for third-party libraries to provide additional information when a Progress Report is generated.  Gomega uses this to provide the current state of an `Eventually().WithContext()` assertion when a Progress Report is requested.
- DescribeTable now exits with an error if it is not passed any Entries [a4c9865]

## Fixes
- fixes crashes on newer Ruby 3 installations by upgrading github-pages gem dependency [92c88d5]
- Make the outline command able to use the DSL import [1be2427]

## Maintenance
- chore(docs): delete no meaning d [57c373c]
- chore(docs): Fix hyperlinks [30526d5]
- chore(docs): fix code blocks without language settings [cf611c4]
- fix intra-doc link [b541bcb]

## 2.2.0

### Generate real-time Progress Reports [f91377c]

Ginkgo can now generate Progress Reports to point users at the current running line of code (including a preview of the actual source code) and a best guess at the most relevant subroutines.

These Progress Reports allow users to debug stuck or slow tests without exiting the Ginkgo process.  A Progress Report can be generated at any time by sending Ginkgo a `SIGINFO` (`^T` on MacOS/BSD) or `SIGUSR1`.

In addition, the user can specify `--poll-progress-after` and `--poll-progress-interval` to have Ginkgo start periodically emitting progress reports if a given node takes too long.  These can be overriden/set on a per-node basis with the `PollProgressAfter` and `PollProgressInterval` decorators.

Progress Reports are emitted to stdout, and also stored in the machine-redable report formats that Ginkgo supports.

Ginkgo also uses this progress reporting infrastructure under the hood when handling timeouts and interrupts.  This yields much more focused, useful, and informative stack traces than previously.

### Features
- `BeforeSuite`, `AfterSuite`, `SynchronizedBeforeSuite`, `SynchronizedAfterSuite`, and `ReportAfterSuite` now support (the relevant subset of) decorators.  These can be passed in _after_ the callback functions that are usually passed into these nodes.

  As a result the **signature of these methods has changed** and now includes a trailing `args ...interface{}`.  For most users simply using the DSL, this change is transparent.  However if you were assigning one of these functions to a custom variable (or passing it around) then your code may need to change to reflect the new signature.

### Maintenance
- Modernize the invocation of Ginkgo in github actions [0ffde58]
- Update reocmmended CI settings in docs [896bbb9]
- Speed up unnecessarily slow integration test [6d3a90e]

## 2.1.6

### Fixes
- Add `SuppressProgressReporting` decorator to turn off --progress announcements for a given node [dfef62a]
- chore: remove duplicate word in comments [7373214]

## 2.1.5

### Fixes
- drop -mod=mod instructions; fixes #1026 [6ad7138]
- Ensure `CurrentSpecReport` and `AddReportEntry` are thread-safe [817c09b]
- remove stale importmap gcflags flag test [3cd8b93]
- Always emit spec summary [5cf23e2] - even when only one spec has failed
- Fix ReportAfterSuite usage in docs [b1864ad]
- fixed typo (#997) [219cc00]
- TrimRight is not designed to trim Suffix [71ebb74]
- refactor: replace strings.Replace with strings.ReplaceAll (#978) [143d208]
- fix syntax in examples (#975) [b69554f]

### Maintenance
- Bump github.com/onsi/gomega from 1.20.0 to 1.20.1 (#1027) [e5dfce4]
- Bump tzinfo from 1.2.9 to 1.2.10 in /docs (#1006) [7ae91c4]
- Bump github.com/onsi/gomega from 1.19.0 to 1.20.0 (#1005) [e87a85a]
- test: add new Go 1.19 to test matrix (#1014) [bbefe12]
- Bump golang.org/x/tools from 0.1.11 to 0.1.12 (#1012) [9327906]
- Bump golang.org/x/tools from 0.1.10 to 0.1.11 (#993) [f44af96]
- Bump nokogiri from 1.13.3 to 1.13.6 in /docs (#981) [ef336aa]

## 2.1.4

### Fixes
- Numerous documentation typos
- Prepend `when` when using `When` (this behavior was in 1.x but unintentionally lost during the 2.0 rewrite) [efce903]
- improve error message when a parallel process fails to report back [a7bd1fe]
- guard against concurrent map writes in DeprecationTracker [0976569]
- Invoke reporting nodes during dry-run (fixes #956 and #935) [aae4480]
- Fix ginkgo import circle [f779385]

## 2.1.3

See [https://onsi.github.io/ginkgo/MIGRATING_TO_V2](https://onsi.github.io/ginkgo/MIGRATING_TO_V2) for details on V2.

### Fixes
- Calling By in a container node now emits a useful error. [ff12cee]

## 2.1.2

### Fixes

- Track location of focused specs correctly in `ginkgo unfocus` [a612ff1]
- Profiling suites with focused specs no longer generates an erroneous failure message [8fbfa02]
- Several documentation typos fixed.  Big thanks to everyone who helped catch them and report/fix them!

## 2.1.1

See [https://onsi.github.io/ginkgo/MIGRATING_TO_V2](https://onsi.github.io/ginkgo/MIGRATING_TO_V2) for details on V2.

### Fixes
- Suites that only import the new dsl packages are now correctly identified as Ginkgo suites [ec17e17]

## 2.1.0

See [https://onsi.github.io/ginkgo/MIGRATING_TO_V2](https://onsi.github.io/ginkgo/MIGRATING_TO_V2) for details on V2.

2.1.0 is a minor release with a few tweaks:

- Introduce new DSL packages to enable users to pick-and-choose which portions of the DSL to dot-import. [90868e2]  More details [here](https://onsi.github.io/ginkgo/#alternatives-to-dot-importing-ginkgo).
- Add error check for invalid/nil parameters to DescribeTable [6f8577e]
- Myriad docs typos fixed (thanks everyone!) [718542a, ecb7098, 146654c, a8f9913, 6bdffde, 03dcd7e]

## 2.0.0

See [https://onsi.github.io/ginkgo/MIGRATING_TO_V2](https://onsi.github.io/ginkgo/MIGRATING_TO_V2)

## 1.16.5

Ginkgo 2.0 now has a Release Candidate.  1.16.5 advertises the existence of the RC.
1.16.5 deprecates GinkgoParallelNode in favor of GinkgoParallelProcess

You can silence the RC advertisement by setting an `ACK_GINKGO_RC=true` environment variable or creating a file in your home directory called `.ack-ginkgo-rc`

## 1.16.4

### Fixes
1.16.4 retracts 1.16.3.  There are no code changes.  The 1.16.3 tag was associated with the wrong commit and an attempt to change it after-the-fact has proven problematic.  1.16.4 retracts 1.16.3 in Ginkgo's go.mod and creates a new, correctly tagged, release.

## 1.16.3

### Features
- Measure is now deprecated and emits a deprecation warning.

## 1.16.2

### Fixes
- Deprecations can be suppressed by setting an `ACK_GINKGO_DEPRECATIONS=<semver>` environment variable.

## 1.16.1

### Fixes
- Suppress --stream deprecation warning on windows (#793)

## 1.16.0

### Features
- Advertise Ginkgo 2.0.  Introduce deprecations. [9ef1913]
    - Update README.md to advertise that Ginkgo 2.0 is coming.
    - Backport the 2.0 DeprecationTracker and start alerting users
    about upcoming deprecations.

- Add slim-sprig template functions to bootstrap/generate (#775) [9162b86]

- Fix accidental reference to 1488 (#784) [9fb7fe4]

## 1.15.2

### Fixes
- ignore blank `-focus` and `-skip` flags (#780) [e90a4a0]

## 1.15.1

### Fixes
- reporters/junit: Use `system-out` element instead of `passed` (#769) [9eda305]

## 1.15.0

### Features
- Adds 'outline' command to print the outline of specs/containers in a file (#754) [071c369] [6803cc3] [935b538] [06744e8] [0c40583]
- Add support for using template to generate tests (#752) [efb9e69]
- Add a Chinese Doc #755 (#756) [5207632]
- cli: allow multiple -focus and -skip flags (#736) [9a782fb]

### Fixes
- Add _internal to filename of tests created with internal flag (#751) [43c12da]

## 1.14.2

### Fixes
- correct handling windows backslash in import path (#721) [97f3d51]
- Add additional methods to GinkgoT() to improve compatibility with the testing.TB interface [b5fe44d]

## 1.14.1

### Fixes
- Discard exported method declaration when running ginkgo bootstrap (#558) [f4b0240]

## 1.14.0

### Features
- Defer running top-level container nodes until RunSpecs is called [d44dedf]
- [Document Ginkgo lifecycle](http://onsi.github.io/ginkgo/#understanding-ginkgos-lifecycle)
- Add `extensions/globals` package (#692) [3295c8f] - this can be helpful in contexts where you are test-driving your test-generation code (see [#692](https://github.com/onsi/ginkgo/pull/692))
- Print Skip reason in JUnit reporter if one was provided [820dfab]

## 1.13.0

### Features
- Add a version of table.Entry that allows dumping the entry parameters. (#689) [21eaef2]

### Fixes
- Ensure integration tests pass in an environment sans GOPATH [606fba2]
- Add books package (#568) [fc0e44e]
- doc(readme): installation via "tools package" (#677) [83bb20e]
- Solve the undefined: unix.Dup2 compile error on mips64le (#680) [0624f75]
- Import package without dot (#687) [6321024]
- Fix integration tests to stop require GOPATH (#686) [a912ec5]

## 1.12.3

### Fixes
- Print correct code location of failing table test (#666) [c6d7afb]

## 1.12.2

### Fixes
- Update dependencies [ea4a036]

## 1.12.1

### Fixes
- Make unfocus ("blur") much faster (#674) [8b18061]
- Fix typo (#673) [7fdcbe8]
- Test against 1.14 and remove 1.12 [d5c2ad6]
- Test if a coverprofile content is empty before checking its latest character (#670) [14d9fa2]
- replace tail package with maintained one. this fixes go get errors (#667) [4ba33d4]
- improve ginkgo performance - makes progress on #644 [a14f98e]
- fix convert integration tests [1f8ba69]
- fix typo successful -> successful (#663) [1ea49cf]
- Fix invalid link (#658) [b886136]
- convert utility : Include comments from source (#657) [1077c6d]
- Explain what BDD means [d79e7fb]
- skip race detector test on unsupported platform (#642) [f8ab89d]
- Use Dup2 from golang.org/x/sys/unix instead of syscallDup (#638) [5d53c55]
- Fix missing newline in combined coverage file (#641) [6a07ea2]
- check if a spec is run before returning SpecSummary (#645) [8850000]

## 1.12.0

### Features
- Add module definition (#630) [78916ab]

## 1.11.0

### Features
- Add syscall for riscv64 architecture [f66e896]
- teamcity reporter: output location of test failure as well as test definition (#626) [9869142]
- teamcity reporter: output newline after every service message (#625) [3cfa02d]
- Add support for go module when running `generate` command (#578) [9c89e3f]

## 1.10.3

### Fixes
- Set go_import_path in travis.yml to allow internal packages in forks (#607) [3b721db]
- Add integration test [d90e0dc]
- Fix coverage files combining [e5dde8c]
- A new CLI option: -ginkgo.reportFile <file path> (#601) [034fd25]

## 1.10.2

### Fixes
- speed up table entry generateIt() (#609) [5049dc5]
- Fix. Write errors to stderr instead of stdout (#610) [7bb3091]

## 1.10.1

### Fixes
- stack backtrace: fix skipping (#600) [2a4c0bd]

## 1.10.0

### Fixes
- stack backtrace: fix alignment and skipping [66915d6]
- fix typo in documentation [8f97b93]

## 1.9.0

### Features
- Option to print output into report, when tests have passed [0545415]

### Fixes
- Fixed typos in comments [0ecbc58]
- gofmt code [a7f8bfb]
- Simplify code [7454d00]
- Simplify concatenation, incrementation and function assignment [4825557]
- Avoid unnecessary conversions [9d9403c]
- JUnit: include more detailed information about panic [19cca4b]
- Print help to stdout when the user asks for help [4cb7441]


## 1.8.0

### New Features
- allow config of the vet flag for `go test` (#562) [3cd45fa]
- Support projects using go modules [d56ee76]

### Fixes and Minor Improvements
- chore(godoc): fixes typos in Measurement funcs [dbaca8e]
- Optimize focus to avoid allocations [f493786]
- Ensure generated test file names are underscored [505cc35]

## 1.7.0

### New Features
- Add JustAfterEach (#484) [0d4f080]

### Fixes
- Correctly round suite time in junit reporter [2445fc1]
- Avoid using -i argument to go test for Golang 1.10+ [46bbc26]

## 1.6.0

### New Features
- add --debug flag to emit node output to files (#499) [39febac]

### Fixes
- fix: for `go vet` to pass [69338ec]
- docs: fix for contributing instructions [7004cb1]
- consolidate and streamline contribution docs (#494) [d848015]
- Make generated Junit file compatible with "Maven Surefire" (#488) [e51bee6]
- all: gofmt [000d317]
- Increase eventually timeout to 30s [c73579c]
- Clarify asynchronous test behavior [294d8f4]
- Travis badge should only show master [26d2143]

## 1.5.0 5/10/2018

### New Features
- Supports go v1.10 (#443, #446, #451) [e873237, 468e89e, e37dbfe, a37f4c0, c0b857d, bca5260, 4177ca8]
- Add a When() synonym for Context() (#386) [747514b, 7484dad, 7354a07, dd826c8]
- Re-add noisySkippings flag [652e15c]
- Allow coverage to be displayed for focused specs (#367) [11459a8]
- Handle -outputdir flag (#364) [228e3a8]
- Handle -coverprofile flag (#355) [43392d5]

### Fixes
- When using custom reporters register the custom reporters *before* the default reporter.  This allows users to see the output of any print statements in their customer reporters. (#365) [8382b23]
- When running a test and calculating the coverage using the `-coverprofile` and `-outputdir` flags, Ginkgo fails with an error if the directory does not exist. This is due to an [issue in go 1.10](https://github.com/golang/go/issues/24588) (#446) [b36a6e0]
- `unfocus` command ignores vendor folder (#459) [e5e551c, c556e43, a3b6351, 9a820dd]
- Ignore packages whose tests are all ignored by go (#456) [7430ca7, 6d8be98]
- Increase the threshold when checking time measurements (#455) [2f714bf, 68f622c]
- Fix race condition in coverage tests (#423) [a5a8ff7, ab9c08b]
- Add an extra new line after reporting spec run completion for test2json [874520d]
- added name name field to junit reported testsuite [ae61c63]
- Do not set the run time of a spec when the dryRun flag is used (#438) [457e2d9, ba8e856]
- Process FWhen and FSpecify when unfocusing (#434) [9008c7b, ee65bd, df87dfe]
- Synchronies the access to the state of specs to avoid race conditions (#430) [7d481bc, ae6829d]
- Added Duration on GinkgoTestDescription (#383) [5f49dad, 528417e, 0747408, 329d7ed]
- Fix Ginkgo stack trace on failure for Specify (#415) [b977ede, 65ca40e, 6c46eb8]
- Update README with Go 1.6+, Golang -> Go (#409) [17f6b97, bc14b66, 20d1598]
- Use fmt.Errorf instead of errors.New(fmt.Sprintf (#401) [a299f56, 44e2eaa]
- Imports in generated code should follow conventions (#398) [0bec0b0, e8536d8]
- Prevent data race error when Recording a benchmark value from multiple go routines (#390) [c0c4881, 7a241e9]
- Replace GOPATH in Environment [4b883f0]


## 1.4.0 7/16/2017

- `ginkgo` now provides a hint if you accidentally forget to run `ginkgo bootstrap` to generate a `*_suite_test.go` file that actually invokes the Ginkgo test runner. [#345](https://github.com/onsi/ginkgo/pull/345)
- thanks to improvements in `go test -c` `ginkgo` no longer needs to fix Go's compilation output to ensure compilation errors are expressed relative to the CWD. [#357]
- `ginkgo watch -watchRegExp=...` allows you to specify a custom regular expression to watch.  Only files matching the regular expression are watched for changes (the default is `\.go$`) [#356]
- `ginkgo` now always emits compilation output.  Previously, only failed compilation output was printed out. [#277]
- `ginkgo -requireSuite` now fails the test run if there are `*_test.go` files but `go test` fails to detect any tests.  Typically this means you forgot to run `ginkgo bootstrap` to generate a suite file. [#344]
- `ginkgo -timeout=DURATION` allows you to adjust the timeout for the entire test suite (default is 24 hours) [#248]

## 1.3.0 3/28/2017

Improvements:

- Significantly improved parallel test distribution.  Now instead of pre-sharding test cases across workers (which can result in idle workers and poor test performance) Ginkgo uses a shared queue to keep all workers busy until all tests are complete.  This improves test-time performance and consistency.
- `Skip(message)` can be used to skip the current test.
- Added `extensions/table` - a Ginkgo DSL for [Table Driven Tests](http://onsi.github.io/ginkgo/#table-driven-tests)
- Add `GinkgoRandomSeed()` - shorthand for `config.GinkgoConfig.RandomSeed`
- Support for retrying flaky tests with `--flakeAttempts`
- `ginkgo ./...` now recurses as you'd expect
- Added `Specify` a synonym for `It`
- Support colorise on Windows
- Broader support for various go compilation flags in the `ginkgo` CLI

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
- When interrupted while concurrently compiling test suites in the background, Ginkgo now cleans up the compiled artifacts.
- Fixed a long standing bug where `ginkgo -p` would hang if a process spawned by one of the Ginkgo parallel nodes does not exit. (Hooray!)

## 1.1.0 (8/2/2014)

No changes, just dropping the beta.

## 1.1.0-beta (7/22/2014)
New Features:

- `ginkgo watch` now monitors packages *and their dependencies* for changes.  The depth of the dependency tree can be modified with the `-depth` flag.
- Test suites with a programmatic focus (`FIt`, `FDescribe`, etc...) exit with non-zero status code, even when they pass.  This allows CI systems to detect accidental commits of focused test suites.
- `ginkgo -p` runs the testsuite in parallel with an auto-detected number of nodes.
- `ginkgo -tags=TAG_LIST` passes a list of tags down to the `go build` command.
- `ginkgo --failFast` aborts the test suite after the first failure.
- `ginkgo generate file_1 file_2` can take multiple file arguments.
- Ginkgo now summarizes any spec failures that occurred at the end of the test run. 
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
