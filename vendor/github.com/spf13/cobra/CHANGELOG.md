# Cobra Changelog

## Pending
* Fix man page doc generation - no auto generated tag when `cmd.DisableAutoGenTag = true` @jpmcb

## v1.0.0
Announcing v1.0.0 of Cobra. 🎉
**Notable Changes**
* Fish completion (including support for Go custom completion) @marckhouzam
* API (urgent): Rename BashCompDirectives to ShellCompDirectives @marckhouzam
* Remove/replace SetOutput on Command - deprecated @jpmcb
* add support for autolabel stale PR @xchapter7x
* Add Labeler Actions @xchapter7x
* Custom completions coded in Go (instead of Bash) @marckhouzam
* Partial Revert of #922 @jharshman
* Add Makefile to project @jharshman
* Correct documentation for InOrStdin @desponda
* Apply formatting to templates @jharshman
* Revert change so help is printed on stdout again @marckhouzam
* Update md2man to v2.0.0 @pdf
* update viper to v1.4.0 @umarcor
* Update cmd/root.go example in README.md @jharshman
