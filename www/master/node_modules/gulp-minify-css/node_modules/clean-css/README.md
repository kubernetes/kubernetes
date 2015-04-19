[![NPM version](https://badge.fury.io/js/clean-css.svg)](https://badge.fury.io/js/clean-css)
[![Build Status](https://secure.travis-ci.org/jakubpawlowicz/clean-css.svg?branch=master)](https://travis-ci.org/jakubpawlowicz/clean-css)
[![Dependency Status](https://david-dm.org/jakubpawlowicz/clean-css.svg?theme=shields.io)](https://david-dm.org/jakubpawlowicz/clean-css)
[![devDependency Status](https://david-dm.org/jakubpawlowicz/clean-css/dev-status.svg?theme=shields.io)](https://david-dm.org/jakubpawlowicz/clean-css#info=devDependencies)

## What is clean-css?

Clean-css is a fast and efficient [Node.js](http://nodejs.org/) library for minifying CSS files.

According to [tests](http://goalsmashers.github.io/css-minification-benchmark/) it is one of the best available.


## Usage

### What are the requirements?

```
Node.js 0.10.0+ (tested on CentOS, Ubuntu, OS X 10.6+, and Windows 7+)
```

### How to install clean-css?

```
npm install clean-css
```

### How to upgrade clean-css from 2.x to 3.x?

#### Command-line interface (CLI) - no breaking changes.

#### Module interface

* `noAdvanced` became `advanced` - make sure to reverse the value;
* `noAggressiveMerging` became `aggressiveMerging` - make sure to reverse the value;
* `noRebase` became `rebase` - make sure to reverse the value;
* no longer possible to use `CleanCSS` as a function as `new CleanCSS` is always required;
* `minify` method returns a hash instead of a string now, so use `new CleanCSS().minify(source).styles` instead of `new CleanCSS().minify(source)`. This change is due to addition of source-maps.
* `stats`, `errors`, and `warnings` are now a properties of a hash returned by `minify` method (see above) instead of CleanCSS instance.

### How to upgrade clean-css from 1.x to 2.x?

#### Command-line interface (CLI)

```
npm update clean-css
```

or point `package.json` to version 2.x. That's it!

#### Node.js module

Update `clean-css` as for CLI above.
Then change your JavaScript code from:

```js
var minimized = CleanCSS.process(source, options);
```

into

```js
var minimized = new CleanCSS(options).minify(source);
```

And you are done.

### How to use clean-css CLI?

Clean-css accepts the following command line arguments (please make sure
you use `<source-file>` as the very last argument to avoid potential issues):

```
cleancss [options] source-file, [source-file, ...]

-h, --help                      Output usage information
-v, --version                   Output the version number
-b, --keep-line-breaks          Keep line breaks
--s0                            Remove all special comments, i.e. /*! comment */
--s1                            Remove all special comments but the first one
-r, --root [root-path]          A root path to which resolve absolute @import rules
                                and rebase relative URLs
-o, --output [output-file]      Use [output-file] as output instead of STDOUT
-s, --skip-import               Disable @import processing
--skip-rebase                   Disable URLs rebasing
--skip-advanced                 Disable advanced optimizations - selector & property merging,
                                reduction, etc.
--skip-aggressive-merging       Disable properties merging based on their order
--skip-shorthand-compacting     Disable shorthand compacting
--rounding-precision [N]        Rounds to `N` decimal places. Defaults to 2. -1 disables rounding.
-c, --compatibility [ie7|ie8]   Force compatibility mode (see Readme for advanced examples)
--source-map                    Enables building input's source map
-d, --debug                     Shows debug information (minification time & compression efficiency)
```

#### Examples:

To minify a **public.css** file into **public-min.css** do:

```
cleancss -o public-min.css public.css
```

To minify the same **public.css** into the standard output skip the `-o` parameter:

```
cleancss public.css
```

More likely you would like to concatenate a couple of files.
If you are on a Unix-like system:

```bash
cat one.css two.css three.css | cleancss -o merged-and-minified.css
```

On Windows:

```bat
type one.css two.css three.css | cleancss -o merged-and-minified.css
```

Or even gzip the result at once:

```bash
cat one.css two.css three.css | cleancss | gzip -9 -c > merged-minified-and-gzipped.css.gz
```

### How to use clean-css programmatically?

```js
var CleanCSS = require('clean-css');
var source = 'a{font-weight:bold;}';
var minimized = new CleanCSS().minify(source).styles;
```

CleanCSS constructor accepts a hash as a parameter, i.e.,
`new CleanCSS(options)` with the following options available:

* `advanced` - set to false to disable advanced optimizations - selector & property merging, reduction, etc.
* `aggressiveMerging` - set to false to disable aggressive merging of properties.
* `benchmark` - turns on benchmarking mode measuring time spent on cleaning up (run `npm run bench` to see example)
* `compatibility` - enables compatibility mode, see [below for more examples](#how-to-set-compatibility-mode)
* `debug` - set to true to get minification statistics under `stats` property (see `test/custom-test.js` for examples)
* `inliner` - a hash of options for `@import` inliner, see test/protocol-imports-test.js for examples
* `keepBreaks` - whether to keep line breaks (default is false)
* `keepSpecialComments` - `*` for keeping all (default), `1` for keeping first one only, `0` for removing all
* `processImport` - whether to process `@import` rules
* `rebase` - set to false to skip URL rebasing
* `relativeTo` - path to __resolve__ relative `@import` rules and URLs
* `root` - path to __resolve__ absolute `@import` rules and __rebase__ relative URLs
* `roundingPrecision` - rounding precision; defaults to `2`; `-1` disables rounding
* `shorthandCompacting` - set to false to skip shorthand compacting (default is true unless sourceMap is set when it's false)
* `sourceMap` - exposes source map under `sourceMap` property, e.g. `new CleanCSS().minify(source).sourceMap` (default is false)
  If input styles are a product of CSS preprocessor (LESS, SASS) an input source map can be passed as a string.
* `target` - path to a folder or an output file to which __rebase__ all URLs

### How to use clean-css with build tools?

* [Broccoli](https://github.com/broccolijs/broccoli#broccoli) : [broccoli-clean-css](https://github.com/shinnn/broccoli-clean-css)
* [Brunch](http://brunch.io/) : [clean-css-brunch](https://github.com/brunch/clean-css-brunch)
* [Grunt](http://gruntjs.com) : [grunt-contrib-cssmin](https://github.com/gruntjs/grunt-contrib-cssmin)
* [Gulp](http://gulpjs.com/) : [gulp-minify-css](https://github.com/jonathanepollack/gulp-minify-css)
* [Gulp](http://gulpjs.com/) : [using vinyl-map as a wrapper - courtesy of @sogko](https://github.com/jakubpawlowicz/clean-css/issues/342)
* [component-builder2](https://github.com/component/builder2.js) : [builder-clean-css](https://github.com/poying/builder-clean-css)
* [Metalsmith](http://metalsmith.io) : [metalsmith-clean-css](https://github.com/aymericbeaumet/metalsmith-clean-css)

### What are the clean-css' dev commands?

First clone the source, then run:

* `npm run bench` for clean-css benchmarks (see [test/bench.js](https://github.com/jakubpawlowicz/clean-css/blob/master/test/bench.js) for details)
* `npm run check` to check JS sources with [JSHint](https://github.com/jshint/jshint/)
* `npm test` for the test suite

## How to contribute to clean-css?

1. Fork it.
2. Add test(s) veryfying the problem.
3. Fix the problem.
4. Make sure all tests still pass (`npm test`).
5. Make sure your code doesn't break style rules (`npm run check`) and follow all [other ones](https://github.com/jakubpawlowicz/clean-css/wiki/Style-Guide) too.
6. Send a PR.

If you wonder where to add tests, go for:

* `test/integration-test.js` if it's a simple scenario
* `test/fixtures/...` if it's a complex scenario (just add two files, input and expected output)
* `test/binary-test.js` if it's related to `bin/cleancss` binary
* `test/module-test.js` if it's related to importing `clean-css` as a module
* `test/protocol-imports-test.js` if it fixes anything related to protocol `@import`s

## Tips & Tricks

### How to preserve a comment block?

Use the `/*!` notation instead of the standard one `/*`:

```css
/*!
  Important comments included in minified output.
*/
```

### How to rebase relative image URLs

Clean-css will handle it automatically for you (since version 1.1) in the following cases:

* When using the CLI:
  1. Use an output path via `-o`/`--output` to rebase URLs as relative to the output file.
  2. Use a root path via `-r`/`--root` to rebase URLs as absolute from the given root path.
  3. If you specify both then `-r`/`--root` takes precendence.
* When using clean-css as a library:
  1. Use a combination of `relativeTo` and `target` options for relative rebase (same as 1 in CLI).
  2. Use a combination of `relativeTo` and `root` options for absolute rebase (same as 2 in CLI).
  3. `root` takes precendence over `target` as in CLI.

### How to generate source maps

Source maps are supported since version 3.0.

Additionally to mapping original CSS files, clean-css also supports input source maps, so minified styles can be mapped into their [LESS](http://lesscss.org/) or [SASS](http://sass-lang.com/) sources directly.

Source maps are generated using [source-map](https://github.com/mozilla/source-map/) module from Mozilla.

#### Using CLI

To generate a source map, use `--source-map` switch, e.g.:

```
cleancss --source-map --output styles.min.css styles.css
```

Name of the output file is required, so a map file, named by adding `.map` suffix to output file name, can be created (styles.min.css.map in this case).

#### Using API

To generate a source map, use `sourceMap: true` option, e.g.:

```javascript
new CleanCSS({ sourceMap: true, target: pathToOutputDirectory }).minify(source, function (minified) {
  // access minified.sourceMap for SourceMapGenerator object
  // see https://github.com/mozilla/source-map/#sourcemapgenerator for more details
  // see https://github.com/jakubpawlowicz/clean-css/blob/master/bin/cleancss#L114 on how it's used in clean-css' CLI
});
```

Using API you can also pass an input source map directly:

```javascript
new CleanCSS({ sourceMap: inputSourceMapAsString, target: pathToOutputDirectory }).minify(source, function (minified) {
  // access minified.sourceMap to access SourceMapGenerator object
  // see https://github.com/mozilla/source-map/#sourcemapgenerator for more details
  // see https://github.com/jakubpawlowicz/clean-css/blob/master/bin/cleancss#L114 on how it's used in clean-css' CLI
});
```

#### Caveats

* Shorthand compacting is currently disabled when source maps are enabled, see [#399](https://github.com/GoalSmashers/clean-css/issues/399)
* Sources inlined in source maps are not supported, see [#397](https://github.com/GoalSmashers/clean-css/issues/397)

### How to set compatibility mode

Compatibility settings are controlled by `--compatibility` switch (CLI) and `compatibility` option (library mode).

In both modes the following values are allowed:

* `'ie7'` - Internet Explorer 7 compatibility mode
* `'ie8'` - Internet Explorer 8 compatibility mode
* `''` or `'*'` (default) - Internet Explorer 9+ compatibility mode

Since clean-css 3 a fine grained control is available over
[those settings](https://github.com/jakubpawlowicz/clean-css/blob/master/lib/utils/compatibility.js),
with the following options available:

* `'[+-]colors.opacity'` - - turn on (+) / off (-) `rgba()` / `hsla()` declarations removal
* `'[+-]properties.iePrefixHack'` - turn on / off IE prefix hack removal
* `'[+-]properties.ieSuffixHack'` - turn on / off IE suffix hack removal
* `'[+-]properties.backgroundSizeMerging'` - turn on / off background-size merging into shorthand
* `'[+-]properties.merging'` - turn on / off property merging based on understandability
* `'[+-]selectors.ie7Hack'` - turn on / off IE7 selector hack removal (`*+html...`)
* `'[+-]units.rem'` - turn on / off treating `rem` as a proper unit

For example, this declaration `--compatibility 'ie8,+units.rem'` will ensure IE8 compatiblity while enabling `rem` units so the following style `margin:0px 0rem` can be shortened to `margin:0`, while in pure IE8 mode it can't be.

To pass a single off (-) switch in CLI please use the following syntax `--compatibility *,-units.rem`.

In library mode you can also pass `compatiblity` as a hash of options.

## Acknowledgments (sorted alphabetically)

* Anthony Barre ([@abarre](https://github.com/abarre)) for improvements to
  `@import` processing, namely introducing the `--skip-import` /
  `processImport` options.
* Simon Altschuler ([@altschuler](https://github.com/altschuler)) for fixing
  `@import` processing inside comments.
* Isaac ([@facelessuser](https://github.com/facelessuser)) for pointing out
  a flaw in clean-css' stateless mode.
* Jan Michael Alonzo ([@jmalonzo](https://github.com/jmalonzo)) for a patch
  removing node.js' old `sys` package.
* Luke Page ([@lukeapage](https://github.com/lukeapage)) for suggestions and testing the source maps feature.
  Plus everyone else involved in [#125](https://github.com/jakubpawlowicz/clean-css/issues/125) for pushing it forward.
* Timur Kristóf ([@Venemo](https://github.com/Venemo)) for an outstanding
  contribution of advanced property optimizer for 2.2 release.
* Vincent Voyer ([@vvo](https://github.com/vvo)) for a patch with better
  empty element regex and for inspiring us to do many performance improvements
  in 0.4 release.
* [@XhmikosR](https://github.com/XhmikosR) for suggesting new features
  (option to remove special comments and strip out URLs quotation) and
  pointing out numerous improvements (JSHint, media queries).

## License

Clean-css is released under the [MIT License](https://github.com/jakubpawlowicz/clean-css/blob/master/LICENSE).
