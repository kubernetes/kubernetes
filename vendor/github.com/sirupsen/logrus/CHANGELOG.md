# 1.4.2
  * Fixes build break for plan9, nacl, solaris
# 1.4.1
This new release introduces:
  * Enhance TextFormatter to not print caller information when they are empty (#944)
  * Remove dependency on golang.org/x/crypto (#932, #943) 

Fixes:
  * Fix Entry.WithContext method to return a copy of the initial entry (#941)

# 1.4.0
This new release introduces:
  * Add `DeferExitHandler`, similar to `RegisterExitHandler` but prepending the handler to the list of handlers (semantically like `defer`) (#848).
  * Add `CallerPrettyfier` to `JSONFormatter` and `TextFormatter (#909, #911)
  * Add `Entry.WithContext()` and `Entry.Context`, to set a context on entries to be used e.g. in hooks (#919).

Fixes:
  * Fix wrong method calls `Logger.Print` and `Logger.Warningln` (#893).
  * Update `Entry.Logf` to not do string formatting unless the log level is enabled (#903)
  * Fix infinite recursion on unknown `Level.String()` (#907)
  * Fix race condition in `getCaller` (#916).


# 1.3.0
This new release introduces:
  * Log, Logf, Logln functions for Logger and Entry that take a Level

Fixes:
  * Building prometheus node_exporter on AIX (#840)
  * Race condition in TextFormatter (#468)
  * Travis CI import path (#868)
  * Remove coloured output on Windows (#862)
  * Pointer to func as field in JSONFormatter (#870)
  * Properly marshal Levels (#873)

# 1.2.0
This new release introduces:
  * A new method `SetReportCaller` in the `Logger` to enable the file, line and calling function from which the trace has been issued
  * A new trace level named `Trace` whose level is below `Debug`
  * A configurable exit function to be called upon a Fatal trace
  * The `Level` object now implements `encoding.TextUnmarshaler` interface

# 1.1.1
This is a bug fix release.
  * fix the build break on Solaris
  * don't drop a whole trace in JSONFormatter when a field param is a function pointer which can not be serialized

# 1.1.0
This new release introduces:
  * several fixes:
    * a fix for a race condition on entry formatting
    * proper cleanup of previously used entries before putting them back in the pool
    * the extra new line at the end of message in text formatter has been removed
  * a new global public API to check if a level is activated: IsLevelEnabled
  * the following methods have been added to the Logger object
    * IsLevelEnabled
    * SetFormatter
    * SetOutput
    * ReplaceHooks
  * introduction of go module
  * an indent configuration for the json formatter
  * output colour support for windows
  * the field sort function is now configurable for text formatter
  * the CLICOLOR and CLICOLOR\_FORCE environment variable support in text formater

# 1.0.6

This new release introduces:
  * a new api WithTime which allows to easily force the time of the log entry
    which is mostly useful for logger wrapper
  * a fix reverting the immutability of the entry given as parameter to the hooks
    a new configuration field of the json formatter in order to put all the fields
    in a nested dictionnary
  * a new SetOutput method in the Logger
  * a new configuration of the textformatter to configure the name of the default keys
  * a new configuration of the text formatter to disable the level truncation

# 1.0.5

* Fix hooks race (#707)
* Fix panic deadlock (#695)

# 1.0.4

* Fix race when adding hooks (#612)
* Fix terminal check in AppEngine (#635)

# 1.0.3

* Replace example files with testable examples

# 1.0.2

* bug: quote non-string values in text formatter (#583)
* Make (*Logger) SetLevel a public method

# 1.0.1

* bug: fix escaping in text formatter (#575)

# 1.0.0

* Officially changed name to lower-case
* bug: colors on Windows 10 (#541)
* bug: fix race in accessing level (#512)

# 0.11.5

* feature: add writer and writerlevel to entry (#372)

# 0.11.4

* bug: fix undefined variable on solaris (#493)

# 0.11.3

* formatter: configure quoting of empty values (#484)
* formatter: configure quoting character (default is `"`) (#484)
* bug: fix not importing io correctly in non-linux environments (#481)

# 0.11.2

* bug: fix windows terminal detection (#476)

# 0.11.1

* bug: fix tty detection with custom out (#471)

# 0.11.0

* performance: Use bufferpool to allocate (#370)
* terminal: terminal detection for app-engine (#343)
* feature: exit handler (#375)

# 0.10.0

* feature: Add a test hook (#180)
* feature: `ParseLevel` is now case-insensitive (#326)
* feature: `FieldLogger` interface that generalizes `Logger` and `Entry` (#308)
* performance: avoid re-allocations on `WithFields` (#335)

# 0.9.0

* logrus/text_formatter: don't emit empty msg
* logrus/hooks/airbrake: move out of main repository
* logrus/hooks/sentry: move out of main repository
* logrus/hooks/papertrail: move out of main repository
* logrus/hooks/bugsnag: move out of main repository
* logrus/core: run tests with `-race`
* logrus/core: detect TTY based on `stderr`
* logrus/core: support `WithError` on logger
* logrus/core: Solaris support

# 0.8.7

* logrus/core: fix possible race (#216)
* logrus/doc: small typo fixes and doc improvements


# 0.8.6

* hooks/raven: allow passing an initialized client

# 0.8.5

* logrus/core: revert #208

# 0.8.4

* formatter/text: fix data race (#218)

# 0.8.3

* logrus/core: fix entry log level (#208)
* logrus/core: improve performance of text formatter by 40%
* logrus/core: expose `LevelHooks` type
* logrus/core: add support for DragonflyBSD and NetBSD
* formatter/text: print structs more verbosely

# 0.8.2

* logrus: fix more Fatal family functions

# 0.8.1

* logrus: fix not exiting on `Fatalf` and `Fatalln`

# 0.8.0

* logrus: defaults to stderr instead of stdout
* hooks/sentry: add special field for `*http.Request`
* formatter/text: ignore Windows for colors

# 0.7.3

* formatter/\*: allow configuration of timestamp layout

# 0.7.2

* formatter/text: Add configuration option for time format (#158)
