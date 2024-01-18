## 1.31.0

### Features
- Async assertions include context cancellation cause if present [121c37f]

### Maintenance
- Bump minimum go version [dee1e3c]
- docs: fix typo in example usage "occured" -> "occurred" [49005fe]
- Bump actions/setup-go from 4 to 5 (#714) [f1c8757]
- Bump github/codeql-action from 2 to 3 (#715) [9836e76]
- Bump github.com/onsi/ginkgo/v2 from 2.13.0 to 2.13.2 (#713) [54726f0]
- Bump golang.org/x/net from 0.17.0 to 0.19.0 (#711) [df97ecc]
- docs: fix `HaveExactElement` typo (#712) [a672c86]

## 1.30.0

### Features
- BeTrueBecause and BeFalseBecause allow for better failure messages [4da4c7f]

### Maintenance
- Bump actions/checkout from 3 to 4 (#694) [6ca6e97]
- doc: fix type on gleak go doc [f1b8343]

## 1.29.0

### Features
- MatchError can now take an optional func(error) bool + description [2b39142]

## 1.28.1

### Maintenance
- Bump github.com/onsi/ginkgo/v2 from 2.12.0 to 2.13.0 [635d196]
- Bump github.com/google/go-cmp from 0.5.9 to 0.6.0 [14f8859]
- Bump golang.org/x/net from 0.14.0 to 0.17.0 [d8a6508]
- #703 doc(matchers): HaveEach() doc comment updated [2705bdb]
- Minor typos (#699) [375648c]

## 1.28.0

### Features
- Add VerifyHost handler to ghttp (#698) [0b03b36]

### Fixes
- Read Body for Newer Responses in HaveHTTPBodyMatcher (#686) [18d6673]

### Maintenance
- Bump github.com/onsi/ginkgo/v2 from 2.11.0 to 2.12.0 (#693) [55a33f3]
- Typo in matchers.go (#691) [de68e8f]
- Bump commonmarker from 0.23.9 to 0.23.10 in /docs (#690) [ab17f5e]
- chore: update test matrix for Go 1.21 (#689) [5069017]
- Bump golang.org/x/net from 0.12.0 to 0.14.0 (#688) [babe25f]

## 1.27.10

### Fixes
- fix: go 1.21 adding goroutine ID to creator+location (#685) [bdc7803]

## 1.27.9

### Fixes
- Prevent nil-dereference in format.Object for boxed nil error (#681) [3b31fc3]

### Maintenance
- Bump golang.org/x/net from 0.11.0 to 0.12.0 (#679) [360849b]
- chore: use String() instead of fmt.Sprintf (#678) [86f3659]
- Bump golang.org/x/net from 0.10.0 to 0.11.0 (#674) [642ead0]
- chore: unnecessary use of fmt.Sprintf (#677) [ceb9ca6]
- Bump github.com/onsi/ginkgo/v2 from 2.10.0 to 2.11.0 (#675) [a2087d8]
- docs: fix ContainSubstring references (#673) [fc9a89f]
- Bump github.com/onsi/ginkgo/v2 from 2.9.7 to 2.10.0 (#671) [9076019]

## 1.27.8

### Fixes
- HaveExactElement should not call FailureMessage if a submatcher returned an error [096f392]

### Maintenance
- Bump github.com/onsi/ginkgo/v2 from 2.9.5 to 2.9.7 (#669) [8884bee]

## 1.27.7

### Fixes
- fix: gcustom.MakeMatcher accepts nil as actual value (#666) [57054d5]

### Maintenance
- update gitignore [05c1bc6]
- Bump github.com/onsi/ginkgo/v2 from 2.9.4 to 2.9.5 (#663) [7cadcf6]
- Bump golang.org/x/net from 0.9.0 to 0.10.0 (#662) [b524839]
- Bump github.com/onsi/ginkgo/v2 from 2.9.2 to 2.9.4 (#661) [5f44694]
- Bump commonmarker from 0.23.8 to 0.23.9 in /docs (#657) [05dc99a]
- Bump nokogiri from 1.14.1 to 1.14.3 in /docs (#658) [3a033d1]
- Replace deprecated NewGomegaWithT with NewWithT (#659) [a19238f]
- Bump golang.org/x/net from 0.8.0 to 0.9.0 (#656) [29ed041]
- Bump actions/setup-go from 3 to 4 (#651) [11b2080]

## 1.27.6

### Fixes
- Allow collections matchers to work correctly when expected has nil elements [60e7cf3]

### Maintenance
- updates MatchError godoc comment to also accept a Gomega matcher (#654) [67b869d]

## 1.27.5

### Maintenance
- Bump github.com/onsi/ginkgo/v2 from 2.9.1 to 2.9.2 (#653) [a215021]
- Bump github.com/go-task/slim-sprig (#652) [a26fed8]

## 1.27.4

### Fixes
- improve error formatting and remove duplication of error message in Eventually/Consistently [854f075]

### Maintenance
- Bump github.com/onsi/ginkgo/v2 from 2.9.0 to 2.9.1 (#650) [ccebd9b]

## 1.27.3

### Fixes
- format.Object now always includes err.Error() when passed an error [86d97ef]
- Fix HaveExactElements to work inside ContainElement or other collection matchers (#648) [636757e]

### Maintenance
- Bump github.com/golang/protobuf from 1.5.2 to 1.5.3 (#649) [cc16689]
- Bump github.com/onsi/ginkgo/v2 from 2.8.4 to 2.9.0 (#646) [e783366]

## 1.27.2

### Fixes
- improve poll progress message when polling a consistently that has been passing [28a319b]

### Maintenance
- bump ginkgo
- remove tools.go hack as Ginkgo 2.8.2 automatically pulls in the cli dependencies [81443b3]

## 1.27.1

### Maintenance

- Bump golang.org/x/net from 0.6.0 to 0.7.0 (#640) [bc686cd]

## 1.27.0

### Features
- Add HaveExactElements matcher (#634) [9d50783]
- update Gomega docs to discuss GinkgoHelper() [be32774]

### Maintenance
- Bump github.com/onsi/ginkgo/v2 from 2.8.0 to 2.8.1 (#639) [296a68b]
- Bump golang.org/x/net from 0.5.0 to 0.6.0 (#638) [c2b098b]
- Bump github-pages from 227 to 228 in /docs (#636) [a9069ab]
- test: update matrix for Go 1.20 (#635) [6bd25c8]
- Bump github.com/onsi/ginkgo/v2 from 2.7.0 to 2.8.0 (#631) [5445f8b]
- Bump webrick from 1.7.0 to 1.8.1 in /docs (#630) [03e93bb]
- codeql: add ruby language (#626) [63c7d21]
- dependabot: add bundler package-ecosystem for docs (#625) [d92f963]

## 1.26.0

### Features
- When a polled function returns an error, keep track of the actual and report on the matcher state of the last non-errored actual [21f3090]
- improve eventually failure message output [c530fb3]

### Fixes
- fix several documentation spelling issues [e2eff1f]


## 1.25.0

### Features
- add `MustPassRepeatedly(int)` to asyncAssertion (#619) [4509f72]
- compare unwrapped errors using DeepEqual (#617) [aaeaa5d]

### Maintenance
- Bump golang.org/x/net from 0.4.0 to 0.5.0 (#614) [c7cfea4]
- Bump github.com/onsi/ginkgo/v2 from 2.6.1 to 2.7.0 (#615) [71b8adb]
- Docs: Fix typo "MUltiple" -> "Multiple" (#616) [9351dda]
- clean up go.sum [cd1dc1d]

## 1.24.2

### Fixes
- Correctly handle assertion failure panics for eventually/consistnetly "g Gomega"s in a goroutine [78f1660]
- docs:Fix typo "you an" -> "you can" (#607) [3187c1f]
- fixes issue #600 (#606) [808d192]

### Maintenance
- Bump golang.org/x/net from 0.2.0 to 0.4.0 (#611) [6ebc0bf]
- Bump nokogiri from 1.13.9 to 1.13.10 in /docs (#612) [258cfc8]
- Bump github.com/onsi/ginkgo/v2 from 2.5.0 to 2.5.1 (#609) [e6c3eb9]

## 1.24.1

### Fixes
- maintain backward compatibility for Eventually and Consisntetly's signatures [4c7df5e]
- fix small typo (#601) [ea0ebe6]

### Maintenance
- Bump golang.org/x/net from 0.1.0 to 0.2.0 (#603) [1ba8372]
- Bump github.com/onsi/ginkgo/v2 from 2.4.0 to 2.5.0 (#602) [f9426cb]
- fix label-filter in test.yml [d795db6]
- stop running flakey tests and rely on external network dependencies in CI [7133290]

## 1.24.0

### Features

Introducting [gcustom](https://onsi.github.io/gomega/#gcustom-a-convenient-mechanism-for-buildling-custom-matchers) - a convenient mechanism for building custom matchers.

This is an RC release for `gcustom`.  The external API may be tweaked in response to feedback however it is expected to remain mostly stable.

### Maintenance

- Update BeComparableTo documentation [756eaa0]

## 1.23.0

### Features
- Custom formatting on a per-type basis can be provided using `format.RegisterCustomFormatter()` -- see the docs [here](https://onsi.github.io/gomega/#adjusting-output)

- Substantial improvement have been made to `StopTrying()`:
  - Users can now use `StopTrying().Wrap(err)` to wrap errors and `StopTrying().Attach(description, object)` to attach arbitrary objects to the `StopTrying()` error
  - `StopTrying()` is now always interpreted as a failure.  If you are an early adopter of `StopTrying()` you may need to change your code as the prior version would match against the returned value even if `StopTrying()` was returned.  Going forward the `StopTrying()` api should remain stable.
  - `StopTrying()` and `StopTrying().Now()` can both be used in matchers - not just polled functions.

- `TryAgainAfter(duration)` is used like `StopTrying()` but instructs `Eventually` and `Consistently` that the poll should be tried again after the specified duration.  This allows you to dynamically adjust the polling duration.

- `ctx` can now be passed-in as the first argument to `Eventually` and `Consistently`.

## Maintenance

- Bump github.com/onsi/ginkgo/v2 from 2.3.0 to 2.3.1 (#597) [afed901]
- Bump nokogiri from 1.13.8 to 1.13.9 in /docs (#599) [7c691b3]
- Bump github.com/google/go-cmp from 0.5.8 to 0.5.9 (#587) [ff22665]

## 1.22.1

## Fixes
- When passed a context and no explicit timeout, Eventually will only timeout when the context is cancelled [e5105cf]
- Allow StopTrying() to be wrapped [bf3cba9]

## Maintenance
- bump to ginkgo v2.3.0 [c5d5c39]

## 1.22.0

### Features

Several improvements have been made to `Eventually` and `Consistently` in this and the most recent releases:

- Eventually and Consistently can take a context.Context [65c01bc]
  This enables integration with Ginkgo 2.3.0's interruptible nodes and node timeouts.
- Eventually and Consistently that are passed a SpecContext can provide reports when an interrupt occurs [0d063c9]
- Eventually/Consistently will forward an attached context to functions that ask for one [e2091c5]
- Eventually/Consistently supports passing arguments to functions via WithArguments() [a2dc7c3]
- Eventually and Consistently can now be stopped early with StopTrying(message) and StopTrying(message).Now() [52976bb]

These improvements are all documented in [Gomega's docs](https://onsi.github.io/gomega/#making-asynchronous-assertions)

## Fixes

## Maintenance

## 1.21.1

### Features
- Eventually and Consistently that are passed a SpecContext can provide reports when an interrupt occurs [0d063c9]

## 1.21.0

### Features
- Eventually and Consistently can take a context.Context [65c01bc]
  This enables integration with Ginkgo 2.3.0's interruptible nodes and node timeouts.
- Introduces Eventually.Within.ProbeEvery with tests and documentation (#591) [f633800]
- New BeKeyOf matcher with documentation and unit tests (#590) [fb586b3]
    
## Fixes
- Cover the entire gmeasure suite with leak detection [8c54344]
- Fix gmeasure leak [119d4ce]
- Ignore new Ginkgo ProgressSignal goroutine in gleak [ba548e2]

## Maintenance

- Fixes crashes on newer Ruby 3 installations by upgrading github-pages gem dependency (#596) [12469a0]


## 1.20.2

## Fixes
- label specs that rely on remote access; bump timeout on short-circuit test to make it less flaky [35eeadf]
- gexec: allow more headroom for SIGABRT-related unit tests (#581) [5b78f40]
- Enable reading from a closed gbytes.Buffer (#575) [061fd26]

## Maintenance
- Bump github.com/onsi/ginkgo/v2 from 2.1.5 to 2.1.6 (#583) [55d895b]
- Bump github.com/onsi/ginkgo/v2 from 2.1.4 to 2.1.5 (#582) [346de7c]

## 1.20.1

## Fixes
- fix false positive gleaks when using ginkgo -p (#577) [cb46517]
- Fix typos in gomega_dsl.go (#569) [5f71ed2]
- don't panic on Eventually(nil), fixing #555 (#567) [9d1186f]
- vet optional description args in assertions, fixing #560 (#566) [8e37808]

## Maintenance
- test: add new Go 1.19 to test matrix (#571) [40d7efe]
- Bump tzinfo from 1.2.9 to 1.2.10 in /docs (#564) [5f26371]

## 1.20.0

## Features
- New [`gleak`](https://onsi.github.io/gomega/#codegleakcode-finding-leaked-goroutines) experimental goroutine leak detection package! (#538) [85ba7bc]
- New `BeComparableTo` matcher(#546) that uses `gocmp` to make comparisons [e77ea75]
- New `HaveExistingField` matcher (#553) [fd130e1]
- Document how to wrap Gomega (#539) [56714a4]

## Fixes
- Support pointer receivers in HaveField; fixes #543 (#544) [8dab36e]

## Maintenance
- Bump various dependencies:
    - Upgrade to yaml.v3 (#556) [f5a83b1]
    - Bump github/codeql-action from 1 to 2 (#549) [52f5adf]
    - Bump github.com/google/go-cmp from 0.5.7 to 0.5.8 (#551) [5f3942d]
    - Bump nokogiri from 1.13.4 to 1.13.6 in /docs (#554) [eb4b4c2]
    - Use latest ginkgo (#535) [1c29028]
    - Bump nokogiri from 1.13.3 to 1.13.4 in /docs (#541) [1ce84d5]
    - Bump actions/setup-go from 2 to 3 (#540) [755485e]
    - Bump nokogiri from 1.12.5 to 1.13.3 in /docs (#522) [4fbb0dc]
    - Bump actions/checkout from 2 to 3 (#526) [ac49202]

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
