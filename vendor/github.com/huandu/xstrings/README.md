# xstrings #

[![Build Status](https://travis-ci.org/huandu/xstrings.svg?branch=master)](https://travis-ci.org/huandu/xstrings)
[![GoDoc](https://godoc.org/github.com/huandu/xstrings?status.svg)](https://godoc.org/github.com/huandu/xstrings)

Go package [xstrings](https://godoc.org/github.com/huandu/xstrings) is a collection of string functions, which are widely used in other languages but absent in Go package [strings](http://golang.org/pkg/strings).

All functions are well tested and carefully tuned for performance.

## Propose a new function ##

Please review [contributing guideline](CONTRIBUTING.md) and [create new issue](https://github.com/huandu/xstrings/issues) to state why it should be included.

## Install ##

Use `go get` to install this library.

	go get github.com/huandu/xstrings

## API document ##

See [GoDoc](https://godoc.org/github.com/huandu/xstrings) for full document.

## Function list ##

Go functions have a unique naming style. One, who has experience in other language but new in Go, may have difficulties to find out right string function to use.

Here is a list of functions in [strings](http://golang.org/pkg/strings) and [xstrings](https://godoc.org/github.com/huandu/xstrings) with enough extra information about how to map these functions to their friends in other languages. Hope this list could be helpful for fresh gophers.

### Package `xstrings` functions ###

*Keep this table sorted by Function in ascending order.*

| Function | Friends | # |
| -------- | ------- | --- |
| [Center](https://godoc.org/github.com/huandu/xstrings#Center) | `str.center` in Python; `String#center` in Ruby | [#30](https://github.com/huandu/xstrings/issues/30) |
| [Count](https://godoc.org/github.com/huandu/xstrings#Count) | `String#count` in Ruby | [#16](https://github.com/huandu/xstrings/issues/16) |
| [Delete](https://godoc.org/github.com/huandu/xstrings#Delete) | `String#delete` in Ruby | [#17](https://github.com/huandu/xstrings/issues/17) |
| [ExpandTabs](https://godoc.org/github.com/huandu/xstrings#ExpandTabs) | `str.expandtabs` in Python | [#27](https://github.com/huandu/xstrings/issues/27) |
| [FirstRuneToLower](https://godoc.org/github.com/huandu/xstrings#FirstRuneToLower) | `lcfirst` in PHP or Perl | [#15](https://github.com/huandu/xstrings/issues/15) |
| [FirstRuneToUpper](https://godoc.org/github.com/huandu/xstrings#FirstRuneToUpper) | `String#capitalize` in Ruby; `ucfirst` in PHP or Perl | [#15](https://github.com/huandu/xstrings/issues/15) |
| [Insert](https://godoc.org/github.com/huandu/xstrings#Insert) | `String#insert` in Ruby | [#18](https://github.com/huandu/xstrings/issues/18) |
| [LastPartition](https://godoc.org/github.com/huandu/xstrings#LastPartition) | `str.rpartition` in Python; `String#rpartition` in Ruby | [#19](https://github.com/huandu/xstrings/issues/19) |
| [LeftJustify](https://godoc.org/github.com/huandu/xstrings#LeftJustify) | `str.ljust` in Python; `String#ljust` in Ruby | [#28](https://github.com/huandu/xstrings/issues/28) |
| [Len](https://godoc.org/github.com/huandu/xstrings#Len) | `mb_strlen` in PHP | [#23](https://github.com/huandu/xstrings/issues/23) |
| [Partition](https://godoc.org/github.com/huandu/xstrings#Partition) | `str.partition` in Python; `String#partition` in Ruby | [#10](https://github.com/huandu/xstrings/issues/10) |
| [Reverse](https://godoc.org/github.com/huandu/xstrings#Reverse) | `String#reverse` in Ruby; `strrev` in PHP; `reverse` in Perl | [#7](https://github.com/huandu/xstrings/issues/7) |
| [RightJustify](https://godoc.org/github.com/huandu/xstrings#RightJustify) | `str.rjust` in Python; `String#rjust` in Ruby | [#29](https://github.com/huandu/xstrings/issues/29) |
| [RuneWidth](https://godoc.org/github.com/huandu/xstrings#RuneWidth) | - | [#27](https://github.com/huandu/xstrings/issues/27) |
| [Scrub](https://godoc.org/github.com/huandu/xstrings#Scrub) | `String#scrub` in Ruby | [#20](https://github.com/huandu/xstrings/issues/20) |
| [Shuffle](https://godoc.org/github.com/huandu/xstrings#Shuffle) | `str_shuffle` in PHP | [#13](https://github.com/huandu/xstrings/issues/13) |
| [ShuffleSource](https://godoc.org/github.com/huandu/xstrings#ShuffleSource) | `str_shuffle` in PHP | [#13](https://github.com/huandu/xstrings/issues/13) |
| [Slice](https://godoc.org/github.com/huandu/xstrings#Slice) | `mb_substr` in PHP | [#9](https://github.com/huandu/xstrings/issues/9) |
| [Squeeze](https://godoc.org/github.com/huandu/xstrings#Squeeze) | `String#squeeze` in Ruby | [#11](https://github.com/huandu/xstrings/issues/11) |
| [Successor](https://godoc.org/github.com/huandu/xstrings#Successor) | `String#succ` or `String#next` in Ruby | [#22](https://github.com/huandu/xstrings/issues/22) |
| [SwapCase](https://godoc.org/github.com/huandu/xstrings#SwapCase) | `str.swapcase` in Python; `String#swapcase` in Ruby | [#12](https://github.com/huandu/xstrings/issues/12) |
| [ToCamelCase](https://godoc.org/github.com/huandu/xstrings#ToCamelCase) | `String#camelize` in RoR | [#1](https://github.com/huandu/xstrings/issues/1) |
| [ToKebab](https://godoc.org/github.com/huandu/xstrings#ToKebabCase) | - | [#41](https://github.com/huandu/xstrings/issues/41) |
| [ToSnakeCase](https://godoc.org/github.com/huandu/xstrings#ToSnakeCase) | `String#underscore` in RoR | [#1](https://github.com/huandu/xstrings/issues/1) |
| [Translate](https://godoc.org/github.com/huandu/xstrings#Translate) | `str.translate` in Python; `String#tr` in Ruby; `strtr` in PHP; `tr///` in Perl | [#21](https://github.com/huandu/xstrings/issues/21) |
| [Width](https://godoc.org/github.com/huandu/xstrings#Width) | `mb_strwidth` in PHP | [#26](https://github.com/huandu/xstrings/issues/26) |
| [WordCount](https://godoc.org/github.com/huandu/xstrings#WordCount) | `str_word_count` in PHP | [#14](https://github.com/huandu/xstrings/issues/14) |
| [WordSplit](https://godoc.org/github.com/huandu/xstrings#WordSplit) | - | [#14](https://github.com/huandu/xstrings/issues/14) |

### Package `strings` functions ###

*Keep this table sorted by Function in ascending order.*

| Function | Friends |
| -------- | ------- |
| [Contains](http://golang.org/pkg/strings/#Contains) | `String#include?` in Ruby |
| [ContainsAny](http://golang.org/pkg/strings/#ContainsAny) | - |
| [ContainsRune](http://golang.org/pkg/strings/#ContainsRune) | - |
| [Count](http://golang.org/pkg/strings/#Count) | `str.count` in Python; `substr_count` in PHP |
| [EqualFold](http://golang.org/pkg/strings/#EqualFold) | `stricmp` in PHP; `String#casecmp` in Ruby |
| [Fields](http://golang.org/pkg/strings/#Fields) | `str.split` in Python; `split` in Perl; `String#split` in Ruby |
| [FieldsFunc](http://golang.org/pkg/strings/#FieldsFunc) | - |
| [HasPrefix](http://golang.org/pkg/strings/#HasPrefix) | `str.startswith` in Python; `String#start_with?` in Ruby |
| [HasSuffix](http://golang.org/pkg/strings/#HasSuffix) | `str.endswith` in Python; `String#end_with?` in Ruby |
| [Index](http://golang.org/pkg/strings/#Index) | `str.index` in Python; `String#index` in Ruby; `strpos` in PHP; `index` in Perl |
| [IndexAny](http://golang.org/pkg/strings/#IndexAny) | - |
| [IndexByte](http://golang.org/pkg/strings/#IndexByte) | - |
| [IndexFunc](http://golang.org/pkg/strings/#IndexFunc) | - |
| [IndexRune](http://golang.org/pkg/strings/#IndexRune) | - |
| [Join](http://golang.org/pkg/strings/#Join) | `str.join` in Python; `Array#join` in Ruby; `implode` in PHP; `join` in Perl |
| [LastIndex](http://golang.org/pkg/strings/#LastIndex) | `str.rindex` in Python; `String#rindex`; `strrpos` in PHP; `rindex` in Perl |
| [LastIndexAny](http://golang.org/pkg/strings/#LastIndexAny) | - |
| [LastIndexFunc](http://golang.org/pkg/strings/#LastIndexFunc) | - |
| [Map](http://golang.org/pkg/strings/#Map) | `String#each_codepoint` in Ruby |
| [Repeat](http://golang.org/pkg/strings/#Repeat) | operator `*` in Python and Ruby; `str_repeat` in PHP |
| [Replace](http://golang.org/pkg/strings/#Replace) | `str.replace` in Python; `String#sub` in Ruby; `str_replace` in PHP |
| [Split](http://golang.org/pkg/strings/#Split) | `str.split` in Python; `String#split` in Ruby; `explode` in PHP; `split` in Perl |
| [SplitAfter](http://golang.org/pkg/strings/#SplitAfter) | - |
| [SplitAfterN](http://golang.org/pkg/strings/#SplitAfterN) | - |
| [SplitN](http://golang.org/pkg/strings/#SplitN) | `str.split` in Python; `String#split` in Ruby; `explode` in PHP; `split` in Perl |
| [Title](http://golang.org/pkg/strings/#Title) | `str.title` in Python |
| [ToLower](http://golang.org/pkg/strings/#ToLower) | `str.lower` in Python; `String#downcase` in Ruby; `strtolower` in PHP; `lc` in Perl |
| [ToLowerSpecial](http://golang.org/pkg/strings/#ToLowerSpecial) | - |
| [ToTitle](http://golang.org/pkg/strings/#ToTitle) | - |
| [ToTitleSpecial](http://golang.org/pkg/strings/#ToTitleSpecial) | - |
| [ToUpper](http://golang.org/pkg/strings/#ToUpper) | `str.upper` in Python; `String#upcase` in Ruby; `strtoupper` in PHP; `uc` in Perl |
| [ToUpperSpecial](http://golang.org/pkg/strings/#ToUpperSpecial) | - |
| [Trim](http://golang.org/pkg/strings/#Trim) | `str.strip` in Python; `String#strip` in Ruby; `trim` in PHP |
| [TrimFunc](http://golang.org/pkg/strings/#TrimFunc) | - |
| [TrimLeft](http://golang.org/pkg/strings/#TrimLeft) | `str.lstrip` in Python; `String#lstrip` in Ruby; `ltrim` in PHP |
| [TrimLeftFunc](http://golang.org/pkg/strings/#TrimLeftFunc) | - |
| [TrimPrefix](http://golang.org/pkg/strings/#TrimPrefix) | - |
| [TrimRight](http://golang.org/pkg/strings/#TrimRight) | `str.rstrip` in Python; `String#rstrip` in Ruby; `rtrim` in PHP |
| [TrimRightFunc](http://golang.org/pkg/strings/#TrimRightFunc) | - |
| [TrimSpace](http://golang.org/pkg/strings/#TrimSpace) | `str.strip` in Python; `String#strip` in Ruby; `trim` in PHP |
| [TrimSuffix](http://golang.org/pkg/strings/#TrimSuffix) | `String#chomp` in Ruby; `chomp` in Perl |

## License ##

This library is licensed under MIT license. See LICENSE for details.
