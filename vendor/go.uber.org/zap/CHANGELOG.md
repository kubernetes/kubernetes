# Changelog
All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.27.0 (20 Feb 2024)
Enhancements:
* [#1378][]: Add `WithLazy` method for `SugaredLogger`.
* [#1399][]: zaptest: Add `NewTestingWriter` for customizing TestingWriter with more flexibility than `NewLogger`.
* [#1406][]: Add `Log`, `Logw`, `Logln` methods for `SugaredLogger`.
* [#1416][]: Add `WithPanicHook` option for testing panic logs.

Thanks to @defval, @dimmo, @arxeiss, and @MKrupauskas for their contributions to this release.

[#1378]: https://github.com/uber-go/zap/pull/1378
[#1399]: https://github.com/uber-go/zap/pull/1399
[#1406]: https://github.com/uber-go/zap/pull/1406
[#1416]: https://github.com/uber-go/zap/pull/1416

## 1.26.0 (14 Sep 2023)
Enhancements:
* [#1297][]: Add Dict as a Field.
* [#1319][]: Add `WithLazy` method to `Logger` which lazily evaluates the structured
context.
* [#1350][]: String encoding is much (~50%) faster now.

Thanks to @hhk7734, @jquirke, and @cdvr1993 for their contributions to this release.

[#1297]: https://github.com/uber-go/zap/pull/1297
[#1319]: https://github.com/uber-go/zap/pull/1319
[#1350]: https://github.com/uber-go/zap/pull/1350

## 1.25.0 (1 Aug 2023)

This release contains several improvements including performance, API additions,
and two new experimental packages whose APIs are unstable and may change in the
future.

Enhancements:
* [#1246][]: Add `zap/exp/zapslog` package for integration with slog.
* [#1273][]: Add `Name` to `Logger` which returns the Logger's name if one is set.
* [#1281][]: Add `zap/exp/expfield` package which contains helper methods
`Str` and `Strs` for constructing String-like zap.Fields.
* [#1310][]: Reduce stack size on `Any`.

Thanks to @knight42, @dzakaammar, @bcspragu, and @rexywork for their contributions
to this release.

[#1246]: https://github.com/uber-go/zap/pull/1246
[#1273]: https://github.com/uber-go/zap/pull/1273
[#1281]: https://github.com/uber-go/zap/pull/1281
[#1310]: https://github.com/uber-go/zap/pull/1310

## 1.24.0 (30 Nov 2022)

Enhancements:
* [#1148][]: Add `Level` to both `Logger` and `SugaredLogger` that reports the
  current minimum enabled log level.
* [#1185][]: `SugaredLogger` turns errors to zap.Error automatically.

Thanks to @Abirdcfly, @craigpastro, @nnnkkk7, and @sashamelentyev for their
contributions to this release.

[#1148]: https://github.coml/uber-go/zap/pull/1148
[#1185]: https://github.coml/uber-go/zap/pull/1185

## 1.23.0 (24 Aug 2022)

Enhancements:
* [#1147][]: Add a `zapcore.LevelOf` function to determine the level of a
  `LevelEnabler` or `Core`.
* [#1155][]: Add `zap.Stringers` field constructor to log arrays of objects
  that implement `String() string`.

[#1147]: https://github.com/uber-go/zap/pull/1147
[#1155]: https://github.com/uber-go/zap/pull/1155

## 1.22.0 (8 Aug 2022)

Enhancements:
* [#1071][]: Add `zap.Objects` and `zap.ObjectValues` field constructors to log
  arrays of objects. With these two constructors, you don't need to implement
  `zapcore.ArrayMarshaler` for use with `zap.Array` if those objects implement
  `zapcore.ObjectMarshaler`.
* [#1079][]: Add `SugaredLogger.WithOptions` to build a copy of an existing
  `SugaredLogger` with the provided options applied.
* [#1080][]: Add `*ln` variants to `SugaredLogger` for each log level.
  These functions provide a string joining behavior similar to `fmt.Println`.
* [#1088][]: Add `zap.WithFatalHook` option to control the behavior of the
  logger for `Fatal`-level log entries. This defaults to exiting the program.
* [#1108][]: Add a `zap.Must` function that you can use with `NewProduction` or
  `NewDevelopment` to panic if the system was unable to build the logger.
* [#1118][]: Add a `Logger.Log` method that allows specifying the log level for
  a statement dynamically.

Thanks to @cardil, @craigpastro, @sashamelentyev, @shota3506, and @zhupeijun
for their contributions to this release.

[#1071]: https://github.com/uber-go/zap/pull/1071
[#1079]: https://github.com/uber-go/zap/pull/1079
[#1080]: https://github.com/uber-go/zap/pull/1080
[#1088]: https://github.com/uber-go/zap/pull/1088
[#1108]: https://github.com/uber-go/zap/pull/1108
[#1118]: https://github.com/uber-go/zap/pull/1118

## 1.21.0 (7 Feb 2022)

Enhancements:
*  [#1047][]: Add `zapcore.ParseLevel` to parse a `Level` from a string.
*  [#1048][]: Add `zap.ParseAtomicLevel` to parse an `AtomicLevel` from a
   string.

Bugfixes:
* [#1058][]: Fix panic in JSON encoder when `EncodeLevel` is unset.

Other changes:
* [#1052][]: Improve encoding performance when the `AddCaller` and
  `AddStacktrace` options are used together.

[#1047]: https://github.com/uber-go/zap/pull/1047
[#1048]: https://github.com/uber-go/zap/pull/1048
[#1052]: https://github.com/uber-go/zap/pull/1052
[#1058]: https://github.com/uber-go/zap/pull/1058

Thanks to @aerosol and @Techassi for their contributions to this release.

## 1.20.0 (4 Jan 2022)

Enhancements:
* [#989][]: Add `EncoderConfig.SkipLineEnding` flag to disable adding newline
  characters between log statements.
* [#1039][]: Add `EncoderConfig.NewReflectedEncoder` field to customize JSON
  encoding of reflected log fields.

Bugfixes:
* [#1011][]: Fix inaccurate precision when encoding complex64 as JSON.
* [#554][], [#1017][]: Close JSON namespaces opened in `MarshalLogObject`
  methods when the methods return.
* [#1033][]: Avoid panicking in Sampler core if `thereafter` is zero.

Other changes:
* [#1028][]: Drop support for Go < 1.15.

[#554]: https://github.com/uber-go/zap/pull/554
[#989]: https://github.com/uber-go/zap/pull/989
[#1011]: https://github.com/uber-go/zap/pull/1011
[#1017]: https://github.com/uber-go/zap/pull/1017
[#1028]: https://github.com/uber-go/zap/pull/1028
[#1033]: https://github.com/uber-go/zap/pull/1033
[#1039]: https://github.com/uber-go/zap/pull/1039

Thanks to @psrajat, @lruggieri, @sammyrnycreal for their contributions to this release.

## 1.19.1 (8 Sep 2021)

Bugfixes:
* [#1001][]: JSON: Fix complex number encoding with negative imaginary part. Thanks to @hemantjadon.
* [#1003][]: JSON: Fix inaccurate precision when encoding float32.

[#1001]: https://github.com/uber-go/zap/pull/1001
[#1003]: https://github.com/uber-go/zap/pull/1003

## 1.19.0 (9 Aug 2021)

Enhancements:
* [#975][]: Avoid panicking in Sampler core if the level is out of bounds.
* [#984][]: Reduce the size of BufferedWriteSyncer by aligning the fields
  better.

[#975]: https://github.com/uber-go/zap/pull/975
[#984]: https://github.com/uber-go/zap/pull/984

Thanks to @lancoLiu and @thockin for their contributions to this release.

## 1.18.1 (28 Jun 2021)

Bugfixes:
* [#974][]: Fix nil dereference in logger constructed by `zap.NewNop`.

[#974]: https://github.com/uber-go/zap/pull/974

## 1.18.0 (28 Jun 2021)

Enhancements:
* [#961][]: Add `zapcore.BufferedWriteSyncer`, a new `WriteSyncer` that buffers
  messages in-memory and flushes them periodically.
* [#971][]: Add `zapio.Writer` to use a Zap logger as an `io.Writer`.
* [#897][]: Add `zap.WithClock` option to control the source of time via the
  new `zapcore.Clock` interface.
* [#949][]: Avoid panicking in `zap.SugaredLogger` when arguments of `*w`
  methods don't match expectations.
* [#943][]: Add support for filtering by level or arbitrary matcher function to
  `zaptest/observer`.
* [#691][]: Comply with `io.StringWriter` and `io.ByteWriter` in Zap's
  `buffer.Buffer`.

Thanks to @atrn0, @ernado, @heyanfu, @hnlq715, @zchee
for their contributions to this release.

[#691]: https://github.com/uber-go/zap/pull/691
[#897]: https://github.com/uber-go/zap/pull/897
[#943]: https://github.com/uber-go/zap/pull/943
[#949]: https://github.com/uber-go/zap/pull/949
[#961]: https://github.com/uber-go/zap/pull/961
[#971]: https://github.com/uber-go/zap/pull/971

## 1.17.0 (25 May 2021)

Bugfixes:
* [#867][]: Encode `<nil>` for nil `error` instead of a panic.
* [#931][], [#936][]: Update minimum version constraints to address
  vulnerabilities in dependencies.

Enhancements:
* [#865][]: Improve alignment of fields of the Logger struct, reducing its
  size from 96 to 80 bytes.
* [#881][]: Support `grpclog.LoggerV2` in zapgrpc.
* [#903][]: Support URL-encoded POST requests to the AtomicLevel HTTP handler
  with the `application/x-www-form-urlencoded` content type.
* [#912][]: Support multi-field encoding with `zap.Inline`.
* [#913][]: Speed up SugaredLogger for calls with a single string.
* [#928][]: Add support for filtering by field name to `zaptest/observer`.

Thanks to @ash2k, @FMLS, @jimmystewpot, @Oncilla, @tsoslow, @tylitianrui, @withshubh, and @wziww for their contributions to this release.

[#865]: https://github.com/uber-go/zap/pull/865
[#867]: https://github.com/uber-go/zap/pull/867
[#881]: https://github.com/uber-go/zap/pull/881
[#903]: https://github.com/uber-go/zap/pull/903
[#912]: https://github.com/uber-go/zap/pull/912
[#913]: https://github.com/uber-go/zap/pull/913
[#928]: https://github.com/uber-go/zap/pull/928
[#931]: https://github.com/uber-go/zap/pull/931
[#936]: https://github.com/uber-go/zap/pull/936

## 1.16.0 (1 Sep 2020)

Bugfixes:
* [#828][]: Fix missing newline in IncreaseLevel error messages.
* [#835][]: Fix panic in JSON encoder when encoding times or durations
  without specifying a time or duration encoder.
* [#843][]: Honor CallerSkip when taking stack traces.
* [#862][]: Fix the default file permissions to use `0666` and rely on the umask instead.
* [#854][]: Encode `<nil>` for nil `Stringer` instead of a panic error log.

Enhancements:
* [#629][]: Added `zapcore.TimeEncoderOfLayout` to easily create time encoders
  for custom layouts.
* [#697][]: Added support for a configurable delimiter in the console encoder.
* [#852][]: Optimize console encoder by pooling the underlying JSON encoder.
* [#844][]: Add ability to include the calling function as part of logs.
* [#843][]: Add `StackSkip` for including truncated stacks as a field.
* [#861][]: Add options to customize Fatal behaviour for better testability.

Thanks to @SteelPhase, @tmshn, @lixingwang, @wyxloading, @moul, @segevfiner, @andy-retailnext and @jcorbin for their contributions to this release.

[#629]: https://github.com/uber-go/zap/pull/629
[#697]: https://github.com/uber-go/zap/pull/697
[#828]: https://github.com/uber-go/zap/pull/828
[#835]: https://github.com/uber-go/zap/pull/835
[#843]: https://github.com/uber-go/zap/pull/843
[#844]: https://github.com/uber-go/zap/pull/844
[#852]: https://github.com/uber-go/zap/pull/852
[#854]: https://github.com/uber-go/zap/pull/854
[#861]: https://github.com/uber-go/zap/pull/861
[#862]: https://github.com/uber-go/zap/pull/862

## 1.15.0 (23 Apr 2020)

Bugfixes:
* [#804][]: Fix handling of `Time` values out of `UnixNano` range.
* [#812][]: Fix `IncreaseLevel` being reset after a call to `With`.

Enhancements:
* [#806][]: Add `WithCaller` option to supersede the `AddCaller` option. This
  allows disabling annotation of log entries with caller information if
  previously enabled with `AddCaller`.
* [#813][]: Deprecate `NewSampler` constructor in favor of
  `NewSamplerWithOptions` which supports a `SamplerHook` option. This option
   adds support for monitoring sampling decisions through a hook.

Thanks to @danielbprice for their contributions to this release.

[#804]: https://github.com/uber-go/zap/pull/804
[#812]: https://github.com/uber-go/zap/pull/812
[#806]: https://github.com/uber-go/zap/pull/806
[#813]: https://github.com/uber-go/zap/pull/813

## 1.14.1 (14 Mar 2020)

Bugfixes:
* [#791][]: Fix panic on attempting to build a logger with an invalid Config.
* [#795][]: Vendoring Zap with `go mod vendor` no longer includes Zap's
  development-time dependencies.
* [#799][]: Fix issue introduced in 1.14.0 that caused invalid JSON output to
  be generated for arrays of `time.Time` objects when using string-based time
  formats.

Thanks to @YashishDua for their contributions to this release.

[#791]: https://github.com/uber-go/zap/pull/791
[#795]: https://github.com/uber-go/zap/pull/795
[#799]: https://github.com/uber-go/zap/pull/799

## 1.14.0 (20 Feb 2020)

Enhancements:
* [#771][]: Optimize calls for disabled log levels.
* [#773][]: Add millisecond duration encoder.
* [#775][]: Add option to increase the level of a logger.
* [#786][]: Optimize time formatters using `Time.AppendFormat` where possible.

Thanks to @caibirdme for their contributions to this release.

[#771]: https://github.com/uber-go/zap/pull/771
[#773]: https://github.com/uber-go/zap/pull/773
[#775]: https://github.com/uber-go/zap/pull/775
[#786]: https://github.com/uber-go/zap/pull/786

## 1.13.0 (13 Nov 2019)

Enhancements:
* [#758][]: Add `Intp`, `Stringp`, and other similar `*p` field constructors
  to log pointers to primitives with support for `nil` values.

Thanks to @jbizzle for their contributions to this release.

[#758]: https://github.com/uber-go/zap/pull/758

## 1.12.0 (29 Oct 2019)

Enhancements:
* [#751][]: Migrate to Go modules.

[#751]: https://github.com/uber-go/zap/pull/751

## 1.11.0 (21 Oct 2019)

Enhancements:
* [#725][]: Add `zapcore.OmitKey` to omit keys in an `EncoderConfig`.
* [#736][]: Add `RFC3339` and `RFC3339Nano` time encoders.

Thanks to @juicemia, @uhthomas for their contributions to this release.

[#725]: https://github.com/uber-go/zap/pull/725
[#736]: https://github.com/uber-go/zap/pull/736

## 1.10.0 (29 Apr 2019)

Bugfixes:
* [#657][]: Fix `MapObjectEncoder.AppendByteString` not adding value as a
  string.
* [#706][]: Fix incorrect call depth to determine caller in Go 1.12.

Enhancements:
* [#610][]: Add `zaptest.WrapOptions` to wrap `zap.Option` for creating test
  loggers.
* [#675][]: Don't panic when encoding a String field.
* [#704][]: Disable HTML escaping for JSON objects encoded using the
  reflect-based encoder.

Thanks to @iaroslav-ciupin, @lelenanam, @joa, @NWilson for their contributions
to this release.

[#657]: https://github.com/uber-go/zap/pull/657
[#706]: https://github.com/uber-go/zap/pull/706
[#610]: https://github.com/uber-go/zap/pull/610
[#675]: https://github.com/uber-go/zap/pull/675
[#704]: https://github.com/uber-go/zap/pull/704

## 1.9.1 (06 Aug 2018)

Bugfixes:

* [#614][]: MapObjectEncoder should not ignore empty slices.

[#614]: https://github.com/uber-go/zap/pull/614

## 1.9.0 (19 Jul 2018)

Enhancements:
* [#602][]: Reduce number of allocations when logging with reflection.
* [#572][], [#606][]: Expose a registry for third-party logging sinks.

Thanks to @nfarah86, @AlekSi, @JeanMertz, @philippgille, @etsangsplk, and
@dimroc for their contributions to this release.

[#602]: https://github.com/uber-go/zap/pull/602
[#572]: https://github.com/uber-go/zap/pull/572
[#606]: https://github.com/uber-go/zap/pull/606

## 1.8.0 (13 Apr 2018)

Enhancements:
* [#508][]: Make log level configurable when redirecting the standard
  library's logger.
* [#518][]: Add a logger that writes to a `*testing.TB`.
* [#577][]: Add a top-level alias for `zapcore.Field` to clean up GoDoc.

Bugfixes:
* [#574][]: Add a missing import comment to `go.uber.org/zap/buffer`.

Thanks to @DiSiqueira and @djui for their contributions to this release.

[#508]: https://github.com/uber-go/zap/pull/508
[#518]: https://github.com/uber-go/zap/pull/518
[#577]: https://github.com/uber-go/zap/pull/577
[#574]: https://github.com/uber-go/zap/pull/574

## 1.7.1 (25 Sep 2017)

Bugfixes:
* [#504][]: Store strings when using AddByteString with the map encoder.

[#504]: https://github.com/uber-go/zap/pull/504

## 1.7.0 (21 Sep 2017)

Enhancements:

* [#487][]: Add `NewStdLogAt`, which extends `NewStdLog` by allowing the user
  to specify the level of the logged messages.

[#487]: https://github.com/uber-go/zap/pull/487

## 1.6.0 (30 Aug 2017)

Enhancements:

* [#491][]: Omit zap stack frames from stacktraces.
* [#490][]: Add a `ContextMap` method to observer logs for simpler
  field validation in tests.

[#490]: https://github.com/uber-go/zap/pull/490
[#491]: https://github.com/uber-go/zap/pull/491

## 1.5.0 (22 Jul 2017)

Enhancements:

* [#460][] and [#470][]: Support errors produced by `go.uber.org/multierr`.
* [#465][]: Support user-supplied encoders for logger names.

Bugfixes:

* [#477][]: Fix a bug that incorrectly truncated deep stacktraces.

Thanks to @richard-tunein and @pavius for their contributions to this release.

[#477]: https://github.com/uber-go/zap/pull/477
[#465]: https://github.com/uber-go/zap/pull/465
[#460]: https://github.com/uber-go/zap/pull/460
[#470]: https://github.com/uber-go/zap/pull/470

## 1.4.1 (08 Jun 2017)

This release fixes two bugs.

Bugfixes:

* [#435][]: Support a variety of case conventions when unmarshaling levels.
* [#444][]: Fix a panic in the observer.

[#435]: https://github.com/uber-go/zap/pull/435
[#444]: https://github.com/uber-go/zap/pull/444

## 1.4.0 (12 May 2017)

This release adds a few small features and is fully backward-compatible.

Enhancements:

* [#424][]: Add a `LineEnding` field to `EncoderConfig`, allowing users to
  override the Unix-style default.
* [#425][]: Preserve time zones when logging times.
* [#431][]: Make `zap.AtomicLevel` implement `fmt.Stringer`, which makes a
  variety of operations a bit simpler.

[#424]: https://github.com/uber-go/zap/pull/424
[#425]: https://github.com/uber-go/zap/pull/425
[#431]: https://github.com/uber-go/zap/pull/431

## 1.3.0 (25 Apr 2017)

This release adds an enhancement to zap's testing helpers as well as the
ability to marshal an AtomicLevel. It is fully backward-compatible.

Enhancements:

* [#415][]: Add a substring-filtering helper to zap's observer. This is
  particularly useful when testing the `SugaredLogger`.
* [#416][]: Make `AtomicLevel` implement `encoding.TextMarshaler`.

[#415]: https://github.com/uber-go/zap/pull/415
[#416]: https://github.com/uber-go/zap/pull/416

## 1.2.0 (13 Apr 2017)

This release adds a gRPC compatibility wrapper. It is fully backward-compatible.

Enhancements:

* [#402][]: Add a `zapgrpc` package that wraps zap's Logger and implements
  `grpclog.Logger`.

[#402]: https://github.com/uber-go/zap/pull/402

## 1.1.0 (31 Mar 2017)

This release fixes two bugs and adds some enhancements to zap's testing helpers.
It is fully backward-compatible.

Bugfixes:

* [#385][]: Fix caller path trimming on Windows.
* [#396][]: Fix a panic when attempting to use non-existent directories with
  zap's configuration struct.

Enhancements:

* [#386][]: Add filtering helpers to zaptest's observing logger.

Thanks to @moitias for contributing to this release.

[#385]: https://github.com/uber-go/zap/pull/385
[#396]: https://github.com/uber-go/zap/pull/396
[#386]: https://github.com/uber-go/zap/pull/386

## 1.0.0 (14 Mar 2017)

This is zap's first stable release. All exported APIs are now final, and no
further breaking changes will be made in the 1.x release series. Anyone using a
semver-aware dependency manager should now pin to `^1`.

Breaking changes:

* [#366][]: Add byte-oriented APIs to encoders to log UTF-8 encoded text without
  casting from `[]byte` to `string`.
* [#364][]: To support buffering outputs, add `Sync` methods to `zapcore.Core`,
  `zap.Logger`, and `zap.SugaredLogger`.
* [#371][]: Rename the `testutils` package to `zaptest`, which is less likely to
  clash with other testing helpers.

Bugfixes:

* [#362][]: Make the ISO8601 time formatters fixed-width, which is friendlier
  for tab-separated console output.
* [#369][]: Remove the automatic locks in `zapcore.NewCore`, which allows zap to
  work with concurrency-safe `WriteSyncer` implementations.
* [#347][]: Stop reporting errors when trying to `fsync` standard out on Linux
  systems.
* [#373][]: Report the correct caller from zap's standard library
  interoperability wrappers.

Enhancements:

* [#348][]: Add a registry allowing third-party encodings to work with zap's
  built-in `Config`.
* [#327][]: Make the representation of logger callers configurable (like times,
  levels, and durations).
* [#376][]: Allow third-party encoders to use their own buffer pools, which
  removes the last performance advantage that zap's encoders have over plugins.
* [#346][]: Add `CombineWriteSyncers`, a convenience function to tee multiple
  `WriteSyncer`s and lock the result.
* [#365][]: Make zap's stacktraces compatible with mid-stack inlining (coming in
  Go 1.9).
* [#372][]: Export zap's observing logger as `zaptest/observer`. This makes it
  easier for particularly punctilious users to unit test their application's
  logging.

Thanks to @suyash, @htrendev, @flisky, @Ulexus, and @skipor for their
contributions to this release.

[#366]: https://github.com/uber-go/zap/pull/366
[#364]: https://github.com/uber-go/zap/pull/364
[#371]: https://github.com/uber-go/zap/pull/371
[#362]: https://github.com/uber-go/zap/pull/362
[#369]: https://github.com/uber-go/zap/pull/369
[#347]: https://github.com/uber-go/zap/pull/347
[#373]: https://github.com/uber-go/zap/pull/373
[#348]: https://github.com/uber-go/zap/pull/348
[#327]: https://github.com/uber-go/zap/pull/327
[#376]: https://github.com/uber-go/zap/pull/376
[#346]: https://github.com/uber-go/zap/pull/346
[#365]: https://github.com/uber-go/zap/pull/365
[#372]: https://github.com/uber-go/zap/pull/372

## 1.0.0-rc.3 (7 Mar 2017)

This is the third release candidate for zap's stable release. There are no
breaking changes.

Bugfixes:

* [#339][]: Byte slices passed to `zap.Any` are now correctly treated as binary blobs
  rather than `[]uint8`.

Enhancements:

* [#307][]: Users can opt into colored output for log levels.
* [#353][]: In addition to hijacking the output of the standard library's
  package-global logging functions, users can now construct a zap-backed
  `log.Logger` instance.
* [#311][]: Frames from common runtime functions and some of zap's internal
  machinery are now omitted from stacktraces.

Thanks to @ansel1 and @suyash for their contributions to this release.

[#339]: https://github.com/uber-go/zap/pull/339
[#307]: https://github.com/uber-go/zap/pull/307
[#353]: https://github.com/uber-go/zap/pull/353
[#311]: https://github.com/uber-go/zap/pull/311

## 1.0.0-rc.2 (21 Feb 2017)

This is the second release candidate for zap's stable release. It includes two
breaking changes.

Breaking changes:

* [#316][]: Zap's global loggers are now fully concurrency-safe
  (previously, users had to ensure that `ReplaceGlobals` was called before the
  loggers were in use). However, they must now be accessed via the `L()` and
  `S()` functions. Users can update their projects with

  ```
  gofmt -r "zap.L -> zap.L()" -w .
  gofmt -r "zap.S -> zap.S()" -w .
  ```
* [#309][] and [#317][]: RC1 was mistakenly shipped with invalid
  JSON and YAML struct tags on all config structs. This release fixes the tags
  and adds static analysis to prevent similar bugs in the future.

Bugfixes:

* [#321][]: Redirecting the standard library's `log` output now
  correctly reports the logger's caller.

Enhancements:

* [#325][] and [#333][]: Zap now transparently supports non-standard, rich
  errors like those produced by `github.com/pkg/errors`.
* [#326][]: Though `New(nil)` continues to return a no-op logger, `NewNop()` is
  now preferred. Users can update their projects with `gofmt -r 'zap.New(nil) ->
  zap.NewNop()' -w .`.
* [#300][]: Incorrectly importing zap as `github.com/uber-go/zap` now returns a
  more informative error.

Thanks to @skipor and @chapsuk for their contributions to this release.

[#316]: https://github.com/uber-go/zap/pull/316
[#309]: https://github.com/uber-go/zap/pull/309
[#317]: https://github.com/uber-go/zap/pull/317
[#321]: https://github.com/uber-go/zap/pull/321
[#325]: https://github.com/uber-go/zap/pull/325
[#333]: https://github.com/uber-go/zap/pull/333
[#326]: https://github.com/uber-go/zap/pull/326
[#300]: https://github.com/uber-go/zap/pull/300

## 1.0.0-rc.1 (14 Feb 2017)

This is the first release candidate for zap's stable release. There are multiple
breaking changes and improvements from the pre-release version. Most notably:

* **Zap's import path is now "go.uber.org/zap"** &mdash; all users will
  need to update their code.
* User-facing types and functions remain in the `zap` package. Code relevant
  largely to extension authors is now in the `zapcore` package.
* The `zapcore.Core` type makes it easy for third-party packages to use zap's
  internals but provide a different user-facing API.
* `Logger` is now a concrete type instead of an interface.
* A less verbose (though slower) logging API is included by default.
* Package-global loggers `L` and `S` are included.
* A human-friendly console encoder is included.
* A declarative config struct allows common logger configurations to be managed
  as configuration instead of code.
* Sampling is more accurate, and doesn't depend on the standard library's shared
  timer heap.

## 0.1.0-beta.1 (6 Feb 2017)

This is a minor version, tagged to allow users to pin to the pre-1.0 APIs and
upgrade at their leisure. Since this is the first tagged release, there are no
backward compatibility concerns and all functionality is new.

Early zap adopters should pin to the 0.1.x minor version until they're ready to
upgrade to the upcoming stable release.
