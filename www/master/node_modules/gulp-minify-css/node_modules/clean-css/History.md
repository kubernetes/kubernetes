[3.0.10 / 2015-02-07](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.9...v3.0.10)
==================

* Fixed issue [#453](https://github.com/jakubpawlowicz/clean-css/issues/453) - double `background-repeat`.
* Fixed issue [#455](https://github.com/jakubpawlowicz/clean-css/issues/455) - property extracting regression.

[3.0.9 / 2015-02-04](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.8...v3.0.9)
==================

* Fixed issue [#452](https://github.com/jakubpawlowicz/clean-css/issues/452) - regression in advanced merging.

[3.0.8 / 2015-01-31](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.7...v3.0.8)
==================

* Fixed issue [#447](https://github.com/GoalSmashers/clean-css/issues/447) - `background-color` in shorthands.
* Fixed issue [#450](https://github.com/GoalSmashers/clean-css/issues/450) - name to hex color converting.

[3.0.7 / 2015-01-22](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.6...v3.0.7)
==================

* Fixed issue [#441](https://github.com/GoalSmashers/clean-css/issues/441) - hex to name color converting.

[3.0.6 / 2015-01-20](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.5...v3.0.6)
==================

* Refixed issue [#414](https://github.com/GoalSmashers/clean-css/issues/414) - source maps position fallback.

[3.0.5 / 2015-01-18](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.4...v3.0.5)
==================

* Fixed issue [#414](https://github.com/GoalSmashers/clean-css/issues/414) - source maps position fallback.
* Fixed issue [#433](https://github.com/GoalSmashers/clean-css/issues/433) - meging `!important` in shorthands.

[3.0.4 / 2015-01-11](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.3...v3.0.4)
==================

* Fixed issue [#314](https://github.com/GoalSmashers/clean-css/issues/314) - spaces inside calc.

[3.0.3 / 2015-01-07](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.2...v3.0.3)
==================

* Just a version bump as npm incorrectly things 2.2.23 is the latest one.

[3.0.2 / 2015-01-04](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.1...v3.0.2)
==================

* Fixed issue [#422](https://github.com/GoalSmashers/clean-css/issues/422) - handling calc as a unit.

[3.0.1 / 2014-12-19](https://github.com/jakubpawlowicz/clean-css/compare/v3.0.0...v3.0.1)
==================

* Fixed issue [#410](https://github.com/GoalSmashers/clean-css/issues/410) - advanced merging and comments.
* Fixed issue [#411](https://github.com/GoalSmashers/clean-css/issues/411) - properties and important comments.

[3.0.0 / 2014-12-18](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.22...v3.0.0)
==================

* Adds more granular control over compatibility settings.
* Adds support for @counter-style at-rule.
* Adds `--source-map`/`sourceMap` switch for building input's source map.
* Adds `--skip-shorthand-compacting`/`shorthandComacting` option for disabling shorthand compacting.
* Allows `target` option to be a path to a folder instead of a file.
* Allows disabling rounding precision. By [@superlukas](https://github.com/superlukas).
* Breaks 2.x compatibility for using CleanCSS as a function.
* Changes `minify` method output to handle multiple outputs.
* Reworks minification to tokenize first then minify.
  See [changes](https://github.com/jakubpawlowicz/clean-css/compare/b06f37d...dd8c14a).
* Removes support for node.js 0.8.x.
* Renames `noAdvanced` option into `advanced`.
* Renames `noAggressiveMerging` option into `aggressiveMerging`.
* Renames `noRebase` option into `rebase`.
* Speeds up advanced processing by shortening optimize loop.
* Fixed issue [#125](https://github.com/GoalSmashers/clean-css/issues/125) - source maps!
* Fixed issue [#344](https://github.com/GoalSmashers/clean-css/issues/344) - merging background-size into shorthand.
* Fixed issue [#352](https://github.com/GoalSmashers/clean-css/issues/352) - honors rebasing in imported stylesheets.
* Fixed issue [#360](https://github.com/GoalSmashers/clean-css/issues/360) - adds 7 extra CSS colors.
* Fixed issue [#363](https://github.com/GoalSmashers/clean-css/issues/363) - `rem` units overriding `px`.
* Fixed issue [#373](https://github.com/GoalSmashers/clean-css/issues/373) - proper background shorthand merging.
* Fixed issue [#395](https://github.com/GoalSmashers/clean-css/issues/395) - unescaped brackets in data URIs.
* Fixed issue [#398](https://github.com/GoalSmashers/clean-css/issues/398) - restoring important comments.
* Fixed issue [#400](https://github.com/GoalSmashers/clean-css/issues/400) - API to accept an array of filenames.
* Fixed issue [#403](https://github.com/GoalSmashers/clean-css/issues/403) - tracking input files in source maps.
* Fixed issue [#404](https://github.com/GoalSmashers/clean-css/issues/404) - no state sharing in API.
* Fixed issue [#405](https://github.com/GoalSmashers/clean-css/issues/405) - disables default background-size merging.
* Refixed issue [#304](https://github.com/GoalSmashers/clean-css/issues/304) - background position merging.

[2.2.22 / 2014-12-13](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.21...v2.2.22)
==================

* Backports fix to issue [#304](https://github.com/GoalSmashers/clean-css/issues/304) - background position merging.

[2.2.21 / 2014-12-10](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.20...v2.2.21)
==================

* Backports fix to issue [#373](https://github.com/GoalSmashers/clean-css/issues/373) - background shorthand merging.

[2.2.20 / 2014-12-02](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.19...v2.2.20)
==================

* Backports fix to issue [#390](https://github.com/GoalSmashers/clean-css/issues/390) - pseudo-class merging.

[2.2.19 / 2014-11-20](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.18...v2.2.19)
==================

* Fixed issue [#385](https://github.com/GoalSmashers/clean-css/issues/385) - edge cases in processing cut off data.

[2.2.18 / 2014-11-17](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.17...v2.2.18)
==================

* Fixed issue [#383](https://github.com/GoalSmashers/clean-css/issues/383) - rounding fractions once again.

[2.2.17 / 2014-11-09](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.16...v2.2.17)
==================

* Fixed issue [#380](https://github.com/GoalSmashers/clean-css/issues/380) - rounding fractions to a whole number.

[2.2.16 / 2014-09-16](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.15...v2.2.16)
==================

* Fixed issue [#359](https://github.com/GoalSmashers/clean-css/issues/359) - handling escaped double backslash.
* Fixed issue [#358](https://github.com/GoalSmashers/clean-css/issues/358) - property merging in compatibility mode.
* Fixed issue [#356](https://github.com/GoalSmashers/clean-css/issues/356) - preserving *+html hack.
* Fixed issue [#354](https://github.com/GoalSmashers/clean-css/issues/354) - !important overriding in shorthands.

[2.2.15 / 2014-09-01](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.14...v2.2.15)
==================

* Fixed issue [#343](https://github.com/GoalSmashers/clean-css/issues/343) - too aggressive rgba/hsla minification.
* Fixed issue [#345](https://github.com/GoalSmashers/clean-css/issues/345) - URL rebasing for document relative ones.
* Fixed issue [#346](https://github.com/GoalSmashers/clean-css/issues/346) - overriding !important by !important.
* Fixed issue [#350](https://github.com/GoalSmashers/clean-css/issues/350) - edge cases in `@import` processing.

[2.2.14 / 2014-08-25](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.13...v2.2.14)
==================

* Makes multival operations idempotent.
* Fixed issue [#339](https://github.com/GoalSmashers/clean-css/issues/339) - skips invalid properties.
* Fixed issue [#341](https://github.com/GoalSmashers/clean-css/issues/341) - ensure output is shorter than input.

[2.2.13 / 2014-08-12](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.12...v2.2.13)

==================

* Fixed issue [#337](https://github.com/jakubpawlowicz/clean-css/issues/337) - handling component importance.

[2.2.12 / 2014-08-02](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.11...v2.2.12)
==================

* Fixed issue with tokenizer removing first selector after an unknown @ rule.
* Fixed issue [#329](https://github.com/jakubpawlowicz/clean-css/issues/329) - font shorthands incorrectly processed.
* Fixed issue [#332](https://github.com/jakubpawlowicz/clean-css/issues/332) - background shorthand with colors.
* Refixed issue [#325](https://github.com/jakubpawlowicz/clean-css/issues/325) - invalid charset declarations.

[2.2.11 / 2014-07-28](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.10...v2.2.11)
==================

* Fixed issue [#326](https://github.com/jakubpawlowicz/clean-css/issues/326) - background-size regression.

[2.2.10 / 2014-07-27](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.9...v2.2.10)
==================

* Improved performance of advanced mode validators.
* Fixed issue [#307](https://github.com/jakubpawlowicz/clean-css/issues/307) - background-color in muliple backgrounds.
* Fixed issue [#322](https://github.com/jakubpawlowicz/clean-css/issues/322) - adds background-size support.
* Fixed issue [#323](https://github.com/jakubpawlowicz/clean-css/issues/323) - stripping variable references.
* Fixed issue [#325](https://github.com/jakubpawlowicz/clean-css/issues/325) - removing invalid @charset declarations.

[2.2.9 / 2014-07-23](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.8...v2.2.9)
==================

* Adds `background` normalization according to W3C spec.
* Fixed issue [#316](https://github.com/jakubpawlowicz/clean-css/issues/316) - incorrect background processing.

[2.2.8 / 2014-07-14](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.7...v2.2.8)
==================

* Fixed issue [#313](https://github.com/jakubpawlowicz/clean-css/issues/313) - processing comment marks in urls.
* Fixed issue [#315](https://github.com/jakubpawlowicz/clean-css/issues/315) - rgba/hsla -> transparent in gradients.

[2.2.7 / 2014-07-10](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.6...v2.2.7)
==================

* Fixed issue [#304](https://github.com/jakubpawlowicz/clean-css/issues/304) - merging multiple backgrounds.
* Fixed issue [#312](https://github.com/jakubpawlowicz/clean-css/issues/312) - merging with mixed repeat.

[2.2.6 / 2014-07-05](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.5...v2.2.6)
==================

* Adds faster quote matching in QuoteScanner.
* Improves QuoteScanner to handle comments correctly.
* Fixed issue [#308](https://github.com/jakubpawlowicz/clean-css/issues/308) - parsing comments in quoted urls.
* Fixed issue [#311](https://github.com/jakubpawlowicz/clean-css/issues/311) - leading/trailing decimal points.

[2.2.5 / 2014-06-29](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.4...v2.2.5)
==================

* Adds removing extra spaces around / in border-radius.
* Adds replacing same horizontal & vertical value in border-radius.
* Fixed issue [#305](https://github.com/jakubpawlowicz/clean-css/issues/305) - allows width keywords in border-width.

[2.2.4 / 2014-06-27](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.3...v2.2.4)
==================

* Fixed issue [#301](https://github.com/jakubpawlowicz/clean-css/issues/301) - proper border radius processing.
* Fixed issue [#303](https://github.com/jakubpawlowicz/clean-css/issues/303) - correctly preserves viewport units.

[2.2.3 / 2014-06-24](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.2...v2.2.3)
==================

* Fixed issue [#302](https://github.com/jakubpawlowicz/clean-css/issues/302) - handling of outline-style: auto.

[2.2.2 / 2014-06-18](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.1...v2.2.2)
==================

* Fixed issue [#297](https://github.com/jakubpawlowicz/clean-css/issues/297) - box-shadow zeros minification.

[2.2.1 / 2014-06-14](https://github.com/jakubpawlowicz/clean-css/compare/v2.2.0...v2.2.1)
==================

* Fixes new property optimizer for 'none' values.
* Fixed issue [#294](https://github.com/jakubpawlowicz/clean-css/issues/294) - space after rgba/hsla in IE<=11.

[2.2.0 / 2014-06-11](https://github.com/jakubpawlowicz/clean-css/compare/v2.1.8...v2.2.0)
==================

* Adds a better algorithm for quotation marks' removal.
* Adds a better non-adjacent optimizer compatible with the upcoming new property optimizer.
* Adds minifying remote files directly from CLI.
* Adds `--rounding-precision` to control rounding precision.
* Moves quotation matching into a `QuoteScanner` class.
* Adds `npm run browserify` for creating embeddable version of clean-css.
* Fixed list-style-* advanced processing.
* Fixed issue [#134](https://github.com/jakubpawlowicz/clean-css/issues/134) - merges properties into shorthand form.
* Fixed issue [#164](https://github.com/jakubpawlowicz/clean-css/issues/164) - removes default values if not needed.
* Fixed issue [#168](https://github.com/jakubpawlowicz/clean-css/issues/168) - adds better property merging algorithm.
* Fixed issue [#173](https://github.com/jakubpawlowicz/clean-css/issues/173) - merges same properties if grouped.
* Fixed issue [#184](https://github.com/jakubpawlowicz/clean-css/issues/184) - uses `!important` for optimization opportunities.
* Fixed issue [#190](https://github.com/jakubpawlowicz/clean-css/issues/190) - uses shorthand to override another shorthand.
* Fixed issue [#197](https://github.com/jakubpawlowicz/clean-css/issues/197) - adds borders merging by understandability.
* Fixed issue [#210](https://github.com/jakubpawlowicz/clean-css/issues/210) - adds temporary workaround for aggressive merging.
* Fixed issue [#246](https://github.com/jakubpawlowicz/clean-css/issues/246) - removes IE hacks when not in compatibility mode.
* Fixed issue [#247](https://github.com/jakubpawlowicz/clean-css/issues/247) - removes deprecated `selectorsMergeMode` switch.
* Refixed issue [#250](https://github.com/jakubpawlowicz/clean-css/issues/250) - based on new quotation marks removal.
* Fixed issue [#257](https://github.com/jakubpawlowicz/clean-css/issues/257) - turns hsla/rgba to transparent if possible.
* Fixed issue [#265](https://github.com/jakubpawlowicz/clean-css/issues/265) - adds support for multiple input files.
* Fixed issue [#275](https://github.com/jakubpawlowicz/clean-css/issues/275) - handling transform properties.
* Fixed issue [#276](https://github.com/jakubpawlowicz/clean-css/issues/276) - corrects unicode handling.
* Fixed issue [#288](https://github.com/jakubpawlowicz/clean-css/issues/288) - adds smarter expression parsing.
* Fixed issue [#293](https://github.com/jakubpawlowicz/clean-css/issues/293) - handles escaped @ symbols in classnames and ids.

[2.1.8 / 2014-03-28](https://github.com/jakubpawlowicz/clean-css/compare/v2.1.7...v2.1.8)
==================

* Fixed issue [#267](https://github.com/jakubpawlowicz/clean-css/issues/267) - incorrect non-adjacent selector merging.

[2.1.7 / 2014-03-24](https://github.com/jakubpawlowicz/clean-css/compare/v2.1.6...v2.1.7)
==================

* Fixed issue [#264](https://github.com/jakubpawlowicz/clean-css/issues/264) - `@import` statements inside comments.

[2.1.6 / 2014-03-10](https://github.com/jakubpawlowicz/clean-css/compare/v2.1.5...v2.1.6)
==================

* Fixed issue [#258](https://github.com/jakubpawlowicz/clean-css/issues/258) - wrong `@import` handling in `EmptyRemoval`.

[2.1.5 / 2014-03-07](https://github.com/jakubpawlowicz/clean-css/compare/v2.1.4...v2.1.5)
==================

* Fixed issue [#255](https://github.com/jakubpawlowicz/clean-css/issues/255) - incorrect processing of a trailing `-0`.

[2.1.4 / 2014-03-01](https://github.com/jakubpawlowicz/clean-css/compare/v2.1.3...v2.1.4)
==================

* Fixed issue [#250](https://github.com/jakubpawlowicz/clean-css/issues/250) - correctly handle JSON data in quotations.

[2.1.3 / 2014-02-26](https://github.com/jakubpawlowicz/clean-css/compare/v2.1.2...v2.1.3)
==================

* Fixed issue [#248](https://github.com/jakubpawlowicz/clean-css/issues/248) - incorrect merging for vendor selectors.

[2.1.2 / 2014-02-25](https://github.com/jakubpawlowicz/clean-css/compare/v2.1.1...v2.1.2)
==================

* Fixed issue [#245](https://github.com/jakubpawlowicz/clean-css/issues/245) - incorrect handling of backslash IE hack.

[2.1.1 / 2014-02-18](https://github.com/jakubpawlowicz/clean-css/compare/v2.1.0...v2.1.1)
==================

* Adds faster selectors processing in advanced optimizer.
* Fixed issue [#241](https://github.com/jakubpawlowicz/clean-css/issues/241) - incorrect handling of `:not()` selectors.

[2.1.0 / 2014-02-13](https://github.com/jakubpawlowicz/clean-css/compare/v2.0.8...v2.1.0)
==================

* Adds an optional callback to minify method.
* Deprecates `--selectors-merge-mode` / `selectorsMergeMode` in favor to `--compatibility` / `compatibility`.
* Fixes debug mode stats for stylesheets using `@import` statements.
* Skips empty removal if advanced processing is enabled.
* Fixed issue [#85](https://github.com/jakubpawlowicz/clean-css/issues/85) - resolving protocol `@import`s.
* Fixed issue [#160](https://github.com/jakubpawlowicz/clean-css/issues/160) - re-runs optimizer until a clean pass.
* Fixed issue [#161](https://github.com/jakubpawlowicz/clean-css/issues/161) - improves tokenizer performance.
* Fixed issue [#163](https://github.com/jakubpawlowicz/clean-css/issues/163) - round pixels to 2nd decimal place.
* Fixed issue [#165](https://github.com/jakubpawlowicz/clean-css/issues/165) - extra space after trailing parenthesis.
* Fixed issue [#186](https://github.com/jakubpawlowicz/clean-css/issues/186) - strip unit from `0rem`.
* Fixed issue [#207](https://github.com/jakubpawlowicz/clean-css/issues/207) - bug in parsing protocol `@import`s.
* Fixed issue [#213](https://github.com/jakubpawlowicz/clean-css/issues/213) - faster rgb to hex transforms.
* Fixed issue [#215](https://github.com/jakubpawlowicz/clean-css/issues/215) - leading zeros in numerical values.
* Fixed issue [#217](https://github.com/jakubpawlowicz/clean-css/issues/217) - whitespace inside attribute selectors and urls.
* Fixed issue [#218](https://github.com/jakubpawlowicz/clean-css/issues/218) - `@import` statements cleanup.
* Fixed issue [#220](https://github.com/jakubpawlowicz/clean-css/issues/220) - selector between comments.
* Fixed issue [#223](https://github.com/jakubpawlowicz/clean-css/issues/223) - two-pass adjacent selectors merging.
* Fixed issue [#226](https://github.com/jakubpawlowicz/clean-css/issues/226) - don't minify `border:none` to `border:0`.
* Fixed issue [#229](https://github.com/jakubpawlowicz/clean-css/issues/229) - improved processing of fraction numbers.
* Fixed issue [#230](https://github.com/jakubpawlowicz/clean-css/issues/230) - better handling of zero values.
* Fixed issue [#235](https://github.com/jakubpawlowicz/clean-css/issues/235) - IE7 compatibility mode.
* Fixed issue [#236](https://github.com/jakubpawlowicz/clean-css/issues/236) - incorrect rebasing with nested `import`s.

[2.0.8 / 2014-02-07](https://github.com/jakubpawlowicz/clean-css/compare/v2.0.7...v2.0.8)
==================

* Fixed issue [#232](https://github.com/jakubpawlowicz/clean-css/issues/232) - edge case in non-adjacent selectors merging.

[2.0.7 / 2014-01-16](https://github.com/jakubpawlowicz/clean-css/compare/v2.0.6...v2.0.7)
==================

* Fixed issue [#208](https://github.com/jakubpawlowicz/clean-css/issues/208) - don't swallow @page and @viewport.

[2.0.6 / 2014-01-04](https://github.com/jakubpawlowicz/clean-css/compare/v2.0.5...v2.0.6)
==================

* Fixed issue [#198](https://github.com/jakubpawlowicz/clean-css/issues/198) - process comments and `@import`s correctly.
* Fixed issue [#205](https://github.com/jakubpawlowicz/clean-css/issues/205) - freeze on broken @import declaration.

[2.0.5 / 2014-01-03](https://github.com/jakubpawlowicz/clean-css/compare/v2.0.4...v2.0.5)
==================

* Fixed issue [#199](https://github.com/jakubpawlowicz/clean-css/issues/199) - keep line breaks with no advanced optimizations.
* Fixed issue [#203](https://github.com/jakubpawlowicz/clean-css/issues/203) - Buffer as a first argument to minify method.

[2.0.4 / 2013-12-19](https://github.com/jakubpawlowicz/clean-css/compare/v2.0.3...v2.0.4)
==================

* Fixed issue [#193](https://github.com/jakubpawlowicz/clean-css/issues/193) - HSL color space normalization.

[2.0.3 / 2013-12-18](https://github.com/jakubpawlowicz/clean-css/compare/v2.0.2...v2.0.3)
==================

* Fixed issue [#191](https://github.com/jakubpawlowicz/clean-css/issues/191) - leading numbers in font/animation names.
* Fixed issue [#192](https://github.com/jakubpawlowicz/clean-css/issues/192) - many `@import`s inside a comment.

[2.0.2 / 2013-11-18](https://github.com/jakubpawlowicz/clean-css/compare/v2.0.1...v2.0.2)
==================

* Fixed issue [#177](https://github.com/jakubpawlowicz/clean-css/issues/177) - process broken content correctly.

[2.0.1 / 2013-11-14](https://github.com/jakubpawlowicz/clean-css/compare/v2.0.0...v2.0.1)
==================

* Fixed issue [#176](https://github.com/jakubpawlowicz/clean-css/issues/176) - hangs on `undefined` keyword.

[2.0.0 / 2013-11-04](https://github.com/jakubpawlowicz/clean-css/compare/v1.1.7...v2.0.0)
==================

* Adds simplified and more advanced text escaping / restoring via `EscapeStore` class.
* Adds simplified and much faster empty elements removal.
* Adds missing `@import` processing to our benchmark (run via `npm run bench`).
* Adds CSS tokenizer which will make it possible to optimize content by reordering and/or merging selectors.
* Adds basic optimizer removing duplicate selectors from a list.
* Adds merging duplicate properties within a single selector's body.
* Adds merging adjacent selectors within a scope (single and multiple ones).
* Changes behavior of `--keep-line-breaks`/`keepBreaks` option to keep breaks after trailing braces only.
* Makes all multiple selectors ordered alphabetically (aids merging).
* Adds property overriding so more coarse properties override more granular ones.
* Adds reducing non-adjacent selectors.
* Adds `--skip-advanced`/`noAdvanced` switch to disable advanced optimizations.
* Adds reducing non-adjacent selectors when overridden by more complex selectors.
* Fixed issue [#138](https://github.com/jakubpawlowicz/clean-css/issues/138) - makes CleanCSS interface OO.
* Fixed issue [#139](https://github.com/jakubpawlowicz/clean-css/issues/138) - consistent error & warning handling.
* Fixed issue [#145](https://github.com/jakubpawlowicz/clean-css/issues/145) - debug mode in library too.
* Fixed issue [#157](https://github.com/jakubpawlowicz/clean-css/issues/157) - gets rid of `removeEmpty` option.
* Fixed issue [#159](https://github.com/jakubpawlowicz/clean-css/issues/159) - escaped quotes inside content.
* Fixed issue [#162](https://github.com/jakubpawlowicz/clean-css/issues/162) - strip quotes from Base64 encoded URLs.
* Fixed issue [#166](https://github.com/jakubpawlowicz/clean-css/issues/166) - `debug` formatting in CLI
* Fixed issue [#167](https://github.com/jakubpawlowicz/clean-css/issues/167) - `background:transparent` minification.

[1.1.7 / 2013-10-28](https://github.com/jakubpawlowicz/clean-css/compare/v1.1.6...v1.1.7)
==================

* Fixed issue [#156](https://github.com/jakubpawlowicz/clean-css/issues/156) - `@import`s inside comments.

[1.1.6 / 2013-10-26](https://github.com/jakubpawlowicz/clean-css/compare/v1.1.5...v1.1.6)
==================

* Fixed issue [#155](https://github.com/jakubpawlowicz/clean-css/issues/155) - broken irregular CSS content.

[1.1.5 / 2013-10-24](https://github.com/jakubpawlowicz/clean-css/compare/v1.1.4...v1.1.5)
==================

* Fixed issue [#153](https://github.com/jakubpawlowicz/clean-css/issues/153) - keepSpecialComments 0/1 as a string.

[1.1.4 / 2013-10-23](https://github.com/jakubpawlowicz/clean-css/compare/v1.1.3...v1.1.4)
==================

* Fixed issue [#152](https://github.com/jakubpawlowicz/clean-css/issues/152) - adds an option to disable rebasing.

[1.1.3 / 2013-10-04](https://github.com/jakubpawlowicz/clean-css/compare/v1.1.2...v1.1.3)
==================

* Fixed issue [#150](https://github.com/jakubpawlowicz/clean-css/issues/150) - minifying `background:none`.

[1.1.2 / 2013-09-29](https://github.com/jakubpawlowicz/clean-css/compare/v1.1.1...v1.1.2)
==================

* Fixed issue [#149](https://github.com/jakubpawlowicz/clean-css/issues/149) - shorthand font property.

[1.1.1 / 2013-09-07](https://github.com/jakubpawlowicz/clean-css/compare/v1.1.0...v1.1.1)
==================

* Fixed issue [#144](https://github.com/jakubpawlowicz/clean-css/issues/144) - skip URLs rebasing by default.

[1.1.0 / 2013-09-06](https://github.com/jakubpawlowicz/clean-css/compare/v1.0.12...v1.1.0)
==================

* Renamed lib's `debug` option to `benchmark` when doing per-minification benchmarking.
* Added simplified comments processing & imports.
* Fixed issue [#43](https://github.com/jakubpawlowicz/clean-css/issues/43) - `--debug` switch for minification stats.
* Fixed issue [#65](https://github.com/jakubpawlowicz/clean-css/issues/65) - full color name / hex shortening.
* Fixed issue [#84](https://github.com/jakubpawlowicz/clean-css/issues/84) - support for `@import` with media queries.
* Fixed issue [#124](https://github.com/jakubpawlowicz/clean-css/issues/124) - raise error on broken imports.
* Fixed issue [#126](https://github.com/jakubpawlowicz/clean-css/issues/126) - proper CSS expressions handling.
* Fixed issue [#129](https://github.com/jakubpawlowicz/clean-css/issues/129) - rebasing imported URLs.
* Fixed issue [#130](https://github.com/jakubpawlowicz/clean-css/issues/130) - better code modularity.
* Fixed issue [#135](https://github.com/jakubpawlowicz/clean-css/issues/135) - require node.js 0.8+.

[1.0.12 / 2013-07-19](https://github.com/jakubpawlowicz/clean-css/compare/v1.0.11...v1.0.12)
===================

* Fixed issue [#121](https://github.com/jakubpawlowicz/clean-css/issues/121) - ability to skip `@import` processing.

[1.0.11 / 2013-07-08](https://github.com/jakubpawlowicz/clean-css/compare/v1.0.10...v1.0.11)
===================

* Fixed issue [#117](https://github.com/jakubpawlowicz/clean-css/issues/117) - line break escaping in comments.

[1.0.10 / 2013-06-13](https://github.com/jakubpawlowicz/clean-css/compare/v1.0.9...v1.0.10)
===================

* Fixed issue [#114](https://github.com/jakubpawlowicz/clean-css/issues/114) - comments in imported stylesheets.

[1.0.9 / 2013-06-11](https://github.com/jakubpawlowicz/clean-css/compare/v1.0.8...v1.0.9)
==================

* Fixed issue [#113](https://github.com/jakubpawlowicz/clean-css/issues/113) - `@import` in comments.

[1.0.8 / 2013-06-10](https://github.com/jakubpawlowicz/clean-css/compare/v1.0.7...v1.0.8)
==================

* Fixed issue [#112](https://github.com/jakubpawlowicz/clean-css/issues/112) - reducing `box-shadow` zeros.

[1.0.7 / 2013-06-05](https://github.com/jakubpawlowicz/clean-css/compare/v1.0.6...v1.0.7)
==================

* Support for `@import` URLs starting with `//`.
  By [@petetak](https://github.com/petetak).

[1.0.6 / 2013-06-04](https://github.com/jakubpawlowicz/clean-css/compare/v1.0.5...v1.0.6)
==================

* Fixed issue [#110](https://github.com/jakubpawlowicz/clean-css/issues/110) - data URIs in URLs.

[1.0.5 / 2013-05-26](https://github.com/jakubpawlowicz/clean-css/compare/v1.0.4...v1.0.5)
==================

* Fixed issue [#107](https://github.com/jakubpawlowicz/clean-css/issues/107) - data URIs in imported stylesheets.

1.0.4 / 2013-05-23
==================

* Rewrite relative URLs in imported stylesheets.
  By [@bluej100](https://github.com/bluej100).

1.0.3 / 2013-05-20
==================

* Support alternative `@import` syntax with file name not wrapped inside `url()` statement.
  By [@bluej100](https://github.com/bluej100).

1.0.2 / 2013-04-29
==================

* Fixed issue [#97](https://github.com/jakubpawlowicz/clean-css/issues/97) - `--remove-empty` & FontAwesome.

1.0.1 / 2013-04-08
==================

* Do not pick up `bench` and `test` while building `npm` package.
  By [@sindresorhus](https://https://github.com/sindresorhus).

1.0.0 / 2013-03-30
==================

* Fixed issue [#2](https://github.com/jakubpawlowicz/clean-css/issues/2) - resolving `@import` rules.
* Fixed issue [#44](https://github.com/jakubpawlowicz/clean-css/issues/44) - examples in `--help`.
* Fixed issue [#46](https://github.com/jakubpawlowicz/clean-css/issues/46) - preserving special characters in URLs and attributes.
* Fixed issue [#80](https://github.com/jakubpawlowicz/clean-css/issues/80) - quotation in multi line strings.
* Fixed issue [#83](https://github.com/jakubpawlowicz/clean-css/issues/83) - HSL to hex color conversions.
* Fixed issue [#86](https://github.com/jakubpawlowicz/clean-css/issues/86) - broken `@charset` replacing.
* Fixed issue [#88](https://github.com/jakubpawlowicz/clean-css/issues/88) - removes space in `! important`.
* Fixed issue [#92](https://github.com/jakubpawlowicz/clean-css/issues/92) - uppercase hex to short versions.

0.10.2 / 2013-03-19
===================

* Fixed issue [#79](https://github.com/jakubpawlowicz/clean-css/issues/79) - node.js 0.10.x compatibility.

0.10.1 / 2013-02-14
===================

* Fixed issue [#66](https://github.com/jakubpawlowicz/clean-css/issues/66) - line breaks without extra spaces should
  be handled correctly.

0.10.0 / 2013-02-09
===================

* Switched from [optimist](https://github.com/substack/node-optimist) to
  [commander](https://github.com/visionmedia/commander.js) for CLI processing.
* Changed long options from `--removeempty` to `--remove-empty` and from `--keeplinebreaks` to `--keep-line-breaks`.
* Fixed performance issue with replacing multiple `@charset` declarations and issue
  with line break after `@charset` when using `keepLineBreaks` option. By [@rrjaime](https://github.com/rrjamie).
* Removed Makefile in favor to `npm run` commands (e.g. `make check` -> `npm run check`).
* Fixed issue [#47](https://github.com/jakubpawlowicz/clean-css/issues/47) - commandline issues on Windows.
* Fixed issue [#49](https://github.com/jakubpawlowicz/clean-css/issues/49) - remove empty selectors from media query.
* Fixed issue [#52](https://github.com/jakubpawlowicz/clean-css/issues/52) - strip fraction zeros if not needed.
* Fixed issue [#58](https://github.com/jakubpawlowicz/clean-css/issues/58) - remove colon where possible.
* Fixed issue [#59](https://github.com/jakubpawlowicz/clean-css/issues/59) - content property handling.

0.9.1 / 2012-12-19
==================

* Fixed issue [#37](https://github.com/jakubpawlowicz/clean-css/issues/37) - converting
  `white` and other colors in class names (reported by [@malgorithms](https://github.com/malgorithms)).

0.9.0 / 2012-12-15
==================

* Added stripping quotation from font names (if possible).
* Added stripping quotation from `@keyframes` declaration, `animation` and
  `animation-name` property.
* Added stripping quotations from attributes' value (e.g. `[data-target='x']`).
* Added better hex->name and name->hex color shortening.
* Added `font: normal` and `font: bold` shortening the same way as `font-weight` is.
* Refactored shorthand selectors and added `border-radius`, `border-style`
  and `border-color` shortening.
* Added `margin`, `padding` and `border-width` shortening.
* Added removing line break after commas.
* Fixed removing whitespace inside media query definition.
* Added removing line breaks after a comma, so all declarations are one-liners now.
* Speed optimizations (~10% despite many new features).
* Added [JSHint](https://github.com/jshint/jshint/) validation rules via `make check`.

0.8.3 / 2012-11-29
==================

* Fixed HSL/HSLA colors processing.

0.8.2 / 2012-10-31
==================

* Fixed shortening hex colors and their relation to hashes in URLs.
* Cleanup by [@XhmikosR](https://github.com/XhmikosR).

0.8.1 / 2012-10-28
==================

* Added better zeros processing for `rect(...)` syntax (clip property).

0.8.0 / 2012-10-21
==================

* Added removing URLs quotation if possible.
* Rewrote breaks processing.
* Added `keepBreaks`/`-b` option to keep line breaks in the minimized file.
* Reformatted [lib/clean.js](/lib/clean.js) so it's easier to follow the rules.
* Minimized test data is now minimized with line breaks so it's easier to
  compare the changes line by line.

0.7.0 / 2012-10-14
==================

* Added stripping special comments to CLI (`--s0` and `--s1` options).
* Added stripping special comments to programmatic interface
  (`keepSpecialComments` option).

0.6.0 / 2012-08-05
==================

* Full Windows support with tests (./test.bat).

0.5.0 / 2012-08-02
==================

* Made path to vows local.
* Explicit node.js 0.6 requirement.

0.4.2 / 2012-06-28
==================

* Updated binary `-v` option (version).
* Updated binary to output help when no options given (but not in piped mode).
* Added binary tests.

0.4.1 / 2012-06-10
==================

* Fixed stateless mode where calling `CleanCSS#process` directly was giving
  errors (reported by [@facelessuser](https://github.com/facelessuser)).

0.4.0 / 2012-06-04
==================

* Speed improvements up to 4x thanks to the rewrite of comments and CSS' content
  processing.
* Stripping empty CSS tags is now optional (see [bin/cleancss](/bin/cleancss) for details).
* Improved debugging mode (see [test/bench.js](/test/bench.js))
* Added `make bench` for a one-pass benchmark.

0.3.3 / 2012-05-27
==================

* Fixed tests, [package.json](/package.json) for development, and regex
  for removing empty declarations (thanks to [@vvo](https://github.com/vvo)).

0.3.2 / 2012-01-17
==================

* Fixed output method under node.js 0.6 which incorrectly tried to close
  `process.stdout`.

0.3.1 / 2011-12-16
==================

* Fixed cleaning up `0 0 0 0` expressions.

0.3.0 / 2011-11-29
==================

* Clean-css requires node.js 0.4.0+ to run.
* Removed node.js's 0.2.x 'sys' package dependency
  (thanks to [@jmalonzo](https://github.com/jmalonzo) for a patch).

0.2.6 / 2011-11-27
==================

* Fixed expanding `+` signs in `calc()` when mixed up with adjacent `+` selector.

0.2.5 / 2011-11-27
==================

* Fixed issue with cleaning up spaces inside `calc`/`-moz-calc` declarations
  (thanks to [@cvan](https://github.com/cvan) for reporting it).
* Fixed converting `#f00` to `red` in borders and gradients.

0.2.4 / 2011-05-25
==================

* Fixed problem with expanding `none` to `0` in partial/full background
  declarations.
* Fixed including clean-css library from binary (global to local).

0.2.3 / 2011-04-18
==================

* Fixed problem with optimizing IE filters.

0.2.2 / 2011-04-17
==================

* Fixed problem with space before color in `border` property.

0.2.1 / 2011-03-19
==================

* Added stripping space before `!important` keyword.
* Updated repository location and author information in [package.json](/package.json).

0.2.0 / 2011-03-02
==================

* Added options parsing via optimist.
* Changed code inclusion (thus the version bump).

0.1.0 / 2011-02-27
==================

* First version of clean-css library.
* Implemented all basic CSS transformations.
