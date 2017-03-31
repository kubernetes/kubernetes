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
