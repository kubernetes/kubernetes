# Cobra Changelog

## v1.1.3

* **Fix:** release-branch.cobra1.1 only: Revert "Deprecate Go < 1.14" to maintain backward compatibility

## v1.1.2

### Notable Changes

* Bump license year to 2021 in golden files (#1309) @Bowbaq
* Enhance PowerShell completion with custom comp (#1208) @Luap99
* Update gopkg.in/yaml.v2 to v2.4.0: The previous breaking change in yaml.v2 v2.3.0 has been reverted, see go-yaml/yaml#670
* Documentation readability improvements (#1228 etc.) @zaataylor etc.
* Use golangci-lint: Repair warnings and errors resulting from linting (#1044) @umarcor

## v1.1.1

* **Fix:** yaml.v2 2.3.0 contained a unintended breaking change. This release reverts to yaml.v2 v2.2.8 which has recent critical CVE fixes, but does not have the breaking changes. See https://github.com/spf13/cobra/pull/1259 for context.
* **Fix:** correct internal formatting for go-md2man v2 (which caused man page generation to be broken). See https://github.com/spf13/cobra/issues/1049 for context.

## v1.1.0

### Notable Changes

* Extend Go completions and revamp zsh comp (#1070)
* Fix man page doc generation - no auto generated tag when `cmd.DisableAutoGenTag = true` (#1104) @jpmcb
* Add completion for help command (#1136)
* Complete subcommands when TraverseChildren is set (#1171)
* Fix stderr printing functions (#894)
* fix: fish output redirection (#1247)

## v1.0.0

Announcing v1.0.0 of Cobra. 🎉

### Notable Changes
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
