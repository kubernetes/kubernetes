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
