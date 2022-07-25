## 1.19.0

## Features
- New [`HaveEach`](https://onsi.github.io/gomega/#haveeachelement-interface) matcher to ensure that each and every element in an `array`, `slice`, or `map` satisfies the passed in matcher. (#523) [9fc2ae2] (#524) [c8ba582]
- Users can now wrap the `Gomega` interface to implement custom behavior on each assertion. (#521) [1f2e714]
- [`ContainElement`](https://onsi.github.io/gomega/#containelementelement-interface) now accepts an additional pointer argument.  Elements that satisfy the matcher are stored in the pointer enabling developers to easily add subsequent, more detailed, assertions against the matching element. (#527) [1a4e27f]

## Fixes
- update RELEASING instructions to match ginkgo [0917cde]
- Bump github.com/onsi/ginkgo/v2 from 2.0.0 to 2.1.3 (#519) [49ab4b0]
- Fix CVE-2021-38561 (#534) [f1b4456]
- Fix max number of samples in experiments on non-64-bit systems. (#528) [1c84497]
- Remove dependency on ginkgo v1.16.4 (#530) [4dea8d5]
- Fix for Go 1.18 (#532) [56d2a29]
- Document precendence of timeouts (#533) [b607941]

## 1.18.1

## Fixes
- Add pointer support to HaveField matcher (#495) [79e41a3]

## 1.18.0

## Features
- Docs now live on the master branch in the docs folder which will make for easier PRs.  The docs also use Ginkgo 2.0's new docs html/css/js. [2570272]
- New HaveValue matcher can handle actuals that are either values (in which case they are passed on unscathed) or pointers (in which case they are indirected).  [Docs here.](https://onsi.github.io/gomega/#working-with-values) (#485) [bdc087c]
- Gmeasure has been declared GA [360db9d]

## Fixes
- Gomega now uses ioutil for Go 1.15 and lower (#492) - official support is only for the most recent two major versions of Go but this will unblock users who need to stay on older unsupported versions of Go. [c29c1c0]

## Maintenace
- Remove Travis workflow (#491) [72e6040]
- Upgrade to Ginkgo 2.0.0 GA [f383637]
- chore: fix description of HaveField matcher (#487) [2b4b2c0]
- use tools.go to ensure Ginkgo cli dependencies are included [f58a52b]
- remove dockerfile and simplify github actions to match ginkgo's actions [3f8160d]

## 1.17.0

### Features
- Add HaveField matcher [3a26311]
- add Error() assertions on the final error value of multi-return values (#480) [2f96943]
- separate out offsets and timeouts (#478) [18a4723]
- fix transformation error reporting (#479) [e001fab]
- allow transform functions to report errors (#472) [bf93408]

### Fixes
Stop using deprecated ioutil package (#467) [07f405d]

## 1.16.0

### Features
- feat: HaveHTTPStatus multiple expected values (#465) [aa69f1b]
- feat: HaveHTTPHeaderWithValue() matcher (#463) [dd83a96]
- feat: HaveHTTPBody matcher (#462) [504e1f2]
- feat: formatter for HTTP responses (#461) [e5b3157]

## 1.15.0

### Fixes
The previous version (1.14.0) introduced a change to allow `Eventually` and `Consistently` to support functions that make assertions.  This was accomplished by overriding the global fail handler when running the callbacks passed to `Eventually/Consistently` in order to capture any resulting errors.  Issue #457 uncovered a flaw with this approach: when multiple `Eventually`s are running concurrently they race when overriding the singleton global fail handler.

1.15.0 resolves this by requiring users who want to make assertions in `Eventually/Consistently` call backs to explicitly pass in a function that takes a `Gomega` as an argument.  The passed-in `Gomega` instance can be used to make assertions.  Any failures will cause `Eventually` to retry the callback.  This cleaner interface avoids the issue of swapping out globals but comes at the cost of changing the contract introduced in v1.14.0.  As such 1.15.0 introduces a breaking change with respect to 1.14.0 - however we expect that adoption of this feature in 1.14.0 remains limited.

In addition, 1.15.0 cleans up some of Gomega's internals.  Most users shouldn't notice any differences stemming from the refactoring that was made.

## 1.14.0

### Features
- gmeasure.SamplingConfig now suppers a MinSamplingInterval [e94dbca]
- Eventually and Consistently support functions that make assertions [2f04e6e]
    - Eventually and Consistently now allow their passed-in functions to make assertions.
    These assertions must pass or the function is considered to have failed and is retried.
    - Eventually and Consistently can now take functions with no return values.  These implicitly return nil
    if they contain no failed assertion.  Otherwise they return an error wrapping the first assertion failure.  This allows
    these functions to be used with the Succeed() matcher.
    - Introduce InterceptGomegaFailure - an analogue to InterceptGomegaFailures - that captures the first assertion failure
    and halts execution in its passed-in callback.

### Fixes
- Call Verify GHTTPWithGomega receiver funcs (#454) [496e6fd]
- Build a binary with an expected name (#446) [7356360]

## 1.13.0

### Features
- gmeasure provides BETA support for benchmarking (#447) [8f2dfbf]
- Set consistently and eventually defaults on init (#443) [12eb778]

## 1.12.0

### Features
- Add Satisfy() matcher (#437) [c548f31]
- tweak truncation message [3360b8c]
- Add format.GomegaStringer (#427) [cc80b6f]
- Add Clear() method to gbytes.Buffer [c3c0920]

### Fixes
- Fix error message in BeNumericallyMatcher (#432) [09c074a]
- Bump github.com/onsi/ginkgo from 1.12.1 to 1.16.2 (#442) [e5f6ea0]
- Bump github.com/golang/protobuf from 1.4.3 to 1.5.2 (#431) [adae3bf]
- Bump golang.org/x/net (#441) [3275b35]

## 1.11.0

### Features
- feature: add index to gstruct element func (#419) [334e00d]
- feat(gexec) Add CompileTest functions. Close #410 (#411) [47c613f]

### Fixes
- Check more carefully for nils in WithTransform (#423) [3c60a15]
- fix: typo in Makefile [b82522a]
- Allow WithTransform function to accept a nil value (#422) [b75d2f2]
- fix: print value type for interface{} containers (#409) [f08e2dc]
- fix(BeElementOf): consistently flatten expected values [1fa9468]

## 1.10.5

### Fixes
- fix: collections matchers should display type of expectation (#408) [6b4eb5a]
- fix(ContainElements): consistently flatten expected values [073b880]
- fix(ConsistOf): consistently flatten expected values [7266efe]

## 1.10.4

### Fixes
- update golang net library to more recent version without vulnerability (#406) [817a8b9]
- Correct spelling: alloted -> allotted (#403) [0bae715]
- fix a panic in MessageWithDiff with long message (#402) [ea06b9b]

## 1.10.3

### Fixes
- updates golang/x/net to fix vulnerability detected by snyk (#394) [c479356]

## 1.10.2

### Fixes
- Add ExpectWithOffset, EventuallyWithOffset and ConsistentlyWithOffset to WithT (#391) [990941a]

## 1.10.1

### Fixes
- Update dependencies (#389) [9f5eecd]

## 1.10.0

### Features
- Add HaveHTTPStatusMatcher (#378) [f335c94]
- Changed matcher for content-type in VerifyJSONRepresenting (#377) [6024f5b]
- Make ghttp usable with x-unit style tests (#376) [c0be499]
- Implement PanicWith matcher (#381) [f8032b4]

## 1.9.0

### Features
- Add ContainElements matcher (#370) [2f57380]
- Output missing and extra elements in ConsistOf failure message [a31eda7]
- Document method LargestMatching [7c5a280]

## 1.8.1

### Fixes
- Fix unexpected MatchError() behaviour (#375) [8ae7b2f]

## 1.8.0

### Features
- Allow optional description to be lazily evaluated function (#364) [bf64010]
- Support wrapped errors (#359) [0a981cb]

## 1.7.1

### Fixes
- Bump go-yaml version to cover fixed ddos heuristic (#362) [95e431e]

## 1.7.0

### Features
- export format property variables (#347) [642e5ba]

### Fixes
- minor fix in the documentation of ExpectWithOffset (#358) [beea727]

## 1.6.0

### Features

- Display special chars on error [41e1b26]
- Add BeElementOf matcher [6a48b48]

### Fixes

- Remove duplication in XML matcher tests [cc1a6cb]
- Remove unnecessary conversions (#357) [7bf756a]
- Fixed import order (#353) [2e3b965]
- Added missing error handling in test (#355) [c98d3eb]
- Simplify code (#356) [0001ed9]
- Simplify code (#354) [0d9100e]
- Fixed typos (#352) [3f647c4]
- Add failure message tests to BeElementOf matcher [efe19c3]
- Update go-testcov untested sections [37ee382]
- Mark all uncovered files so go-testcov ./... works [53b150e]
- Reenable gotip in travis [5c249dc]
- Fix the typo of comment (#345) [f0e010e]
- Optimize contain_element_matcher [abeb93d]


## 1.5.0

### Features

- Added MatchKeys matchers [8b909fc]

### Fixes and Minor Improvements

- Add type aliases to remove stuttering [03b0461]
- Don't run session_test.go on windows (#324) [5533ce8]

## 1.4.3

### Fixes:

- ensure file name and line numbers are correctly reported for XUnit [6fff58f]
- Fixed matcher for content-type (#305) [69d9b43]

## 1.4.2

### Fixes:

- Add go.mod and go.sum files to define the gomega go module [f3de367, a085d30]
- Work around go vet issue with Go v1.11 (#300) [40dd6ad]
- Better output when using with go XUnit-style tests, fixes #255 (#297) [29a4b97]
- Fix MatchJSON fail to parse json.RawMessage (#298) [ae19f1b]
- show threshold in failure message of BeNumericallyMatcher (#293) [4bbecc8]

## 1.4.1

### Fixes:

- Update documentation formatting and examples (#289) [9be8410]
- allow 'Receive' matcher to be used with concrete types (#286) [41673fd]
- Fix data race in ghttp server (#283) [7ac6b01]
- Travis badge should only show master [cc102ab]

## 1.4.0

### Features
- Make string pretty diff user configurable (#273) [eb112ce, 649b44d]

### Fixes
- Use httputil.DumpRequest to pretty-print unhandled requests (#278) [a4ff0fc, b7d1a52]
- fix typo floa32 > float32 (#272) [041ae3b, 6e33911]
- Fix link to documentation on adding your own matchers (#270) [bb2c830, fcebc62]
- Use setters and getters to avoid race condition (#262) [13057c3, a9c79f1]
- Avoid sending a signal if the process is not alive (#259) [b8043e5, 4fc1762]
- Improve message from AssignableToTypeOf when expected value is nil (#281) [9c1fb20]

## 1.3.0

Improvements:

- The `Equal` matcher matches byte slices more performantly.
- Improved how `MatchError` matches error strings.
- `MatchXML` ignores the order of xml node attributes.
- Improve support for XUnit style golang tests. ([#254](https://github.com/onsi/gomega/issues/254))

Bug Fixes:

- Diff generation now handles multi-byte sequences correctly.
- Multiple goroutines can now call `gexec.Build` concurrently.

## 1.2.0

Improvements:

- Added `BeSent` which attempts to send a value down a channel and fails if the attempt blocks.  Can be paired with `Eventually` to safely send a value down a channel with a timeout.
- `Ω`, `Expect`, `Eventually`, and `Consistently` now immediately `panic` if there is no registered fail handler.  This is always a mistake that can hide failing tests.
- `Receive()` no longer errors when passed a closed channel, it's perfectly fine to attempt to read from a closed channel so Ω(c).Should(Receive()) always fails and Ω(c).ShoudlNot(Receive()) always passes with a closed channel.
- Added `HavePrefix` and `HaveSuffix` matchers.
- `ghttp` can now handle concurrent requests.
- Added `Succeed` which allows one to write `Ω(MyFunction()).Should(Succeed())`.
- Improved `ghttp`'s behavior around failing assertions and panics:
    - If a registered handler makes a failing assertion `ghttp` will return `500`.
    - If a registered handler panics, `ghttp` will return `500` *and* fail the test.  This is new behavior that may cause existing code to break.  This code is almost certainly incorrect and creating a false positive.
- `ghttp` servers can take an `io.Writer`.  `ghttp` will write a line to the writer when each request arrives.
- Added `WithTransform` matcher to allow munging input data before feeding into the relevant matcher
- Added boolean `And`, `Or`, and `Not` matchers to allow creating composite matchers
- Added `gbytes.TimeoutCloser`, `gbytes.TimeoutReader`, and `gbytes.TimeoutWriter` - these are convenience wrappers that timeout if the underlying Closer/Reader/Writer does not return within the alloted time.
- Added `gbytes.BufferReader` - this constructs a `gbytes.Buffer` that asynchronously reads the passed-in `io.Reader` into its buffer.

Bug Fixes:
- gexec: `session.Wait` now uses `EventuallyWithOffset` to get the right line number in the failure.
- `ContainElement` no longer bails if a passed-in matcher errors.

## 1.0 (8/2/2014)

No changes. Dropping "beta" from the version number.

## 1.0.0-beta (7/8/2014)
Breaking Changes:

- Changed OmegaMatcher interface.  Instead of having `Match` return failure messages, two new methods `FailureMessage` and `NegatedFailureMessage` are called instead.
- Moved and renamed OmegaFailHandler to types.GomegaFailHandler and OmegaMatcher to types.GomegaMatcher.  Any references to OmegaMatcher in any custom matchers will need to be changed to point to types.GomegaMatcher

New Test-Support Features:

- `ghttp`: supports testing http clients
    - Provides a flexible fake http server
    - Provides a collection of chainable http handlers that perform assertions.
- `gbytes`: supports making ordered assertions against streams of data
    - Provides a `gbytes.Buffer`
    - Provides a `Say` matcher to perform ordered assertions against output data
- `gexec`: supports testing external processes
    - Provides support for building Go binaries
    - Wraps and starts `exec.Cmd` commands
    - Makes it easy to assert against stdout and stderr
    - Makes it easy to send signals and wait for processes to exit
    - Provides an `Exit` matcher to assert against exit code.

DSL Changes:

- `Eventually` and `Consistently` can accept `time.Duration` interval and polling inputs.
- The default timeouts for `Eventually` and `Consistently` are now configurable.

New Matchers:

- `ConsistOf`: order-independent assertion against the elements of an array/slice or keys of a map.
- `BeTemporally`: like `BeNumerically` but for `time.Time`
- `HaveKeyWithValue`: asserts a map has a given key with the given value.

Updated Matchers:

- `Receive` matcher can take a matcher as an argument and passes only if the channel under test receives an objet that satisfies the passed-in matcher.
- Matchers that implement `MatchMayChangeInTheFuture(actual interface{}) bool` can inform `Eventually` and/or `Consistently` when a match has no chance of changing status in the future.  For example, `Receive` returns `false` when a channel is closed.

Misc:

- Start using semantic versioning
- Start maintaining changelog

Major refactor:

- Pull out Gomega's internal to `internal`
