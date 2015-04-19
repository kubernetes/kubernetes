
2.1.3 / 2015-03-13
==================

  * Updated stdout/stderr example (#186)
  * Updated example/stdout.js to match debug current behaviour
  * Renamed example/stderr.js to stdout.js
  * Update Readme.md (#184)
  * replace high intensity foreground color for bold (#182, #183)

2.1.2 / 2015-03-01
==================

  * dist: recompile
  * update "ms" to v0.7.0
  * package: update "browserify" to v9.0.3
  * component: fix "ms.js" repo location
  * changed bower package name
  * updated documentation about using debug in a browser
  * fix: security error on safari (#167, #168, @yields)

2.1.1 / 2014-12-29
==================

  * browser: use `typeof` to check for `console` existence
  * browser: check for `console.log` truthiness (fix IE 8/9)
  * browser: add support for Chrome apps
  * Readme: added Windows usage remarks
  * Add `bower.json` to properly support bower install

2.1.0 / 2014-10-15
==================

  * node: implement `DEBUG_FD` env variable support
  * package: update "browserify" to v6.1.0
  * package: add "license" field to package.json (#135, @panuhorsmalahti)

2.0.0 / 2014-09-01
==================

  * package: update "browserify" to v5.11.0
  * node: use stderr rather than stdout for logging (#29, @stephenmathieson)

1.0.4 / 2014-07-15
==================

  * dist: recompile
  * example: remove `console.info()` log usage
  * example: add "Content-Type" UTF-8 header to browser example
  * browser: place %c marker after the space character
  * browser: reset the "content" color via `color: inherit`
  * browser: add colors support for Firefox >= v31
  * debug: prefer an instance `log()` function over the global one (#119)
  * Readme: update documentation about styled console logs for FF v31 (#116, @wryk)

1.0.3 / 2014-07-09
==================

  * Add support for multiple wildcards in namespaces (#122, @seegno)
  * browser: fix lint

1.0.2 / 2014-06-10
==================

  * browser: update color palette (#113, @gscottolson)
  * common: make console logging function configurable (#108, @timoxley)
  * node: fix %o colors on old node <= 0.8.x
  * Makefile: find node path using shell/which (#109, @timoxley)

1.0.1 / 2014-06-06
==================

  * browser: use `removeItem()` to clear localStorage
  * browser, node: don't set DEBUG if namespaces is undefined (#107, @leedm777)
  * package: add "contributors" section
  * node: fix comment typo
  * README: list authors

1.0.0 / 2014-06-04
==================

  * make ms diff be global, not be scope
  * debug: ignore empty strings in enable()
  * node: make DEBUG_COLORS able to disable coloring
  * *: export the `colors` array
  * npmignore: don't publish the `dist` dir
  * Makefile: refactor to use browserify
  * package: add "browserify" as a dev dependency
  * Readme: add Web Inspector Colors section
  * node: reset terminal color for the debug content
  * node: map "%o" to `util.inspect()`
  * browser: map "%j" to `JSON.stringify()`
  * debug: add custom "formatters"
  * debug: use "ms" module for humanizing the diff
  * Readme: add "bash" syntax highlighting
  * browser: add Firebug color support
  * browser: add colors for WebKit browsers
  * node: apply log to `console`
  * rewrite: abstract common logic for Node & browsers
  * add .jshintrc file

0.8.1 / 2014-04-14
==================

  * package: re-add the "component" section

0.8.0 / 2014-03-30
==================

  * add `enable()` method for nodejs. Closes #27
  * change from stderr to stdout
  * remove unnecessary index.js file

0.7.4 / 2013-11-13
==================

  * remove "browserify" key from package.json (fixes something in browserify)

0.7.3 / 2013-10-30
==================

  * fix: catch localStorage security error when cookies are blocked (Chrome)
  * add debug(err) support. Closes #46
  * add .browser prop to package.json. Closes #42

0.7.2 / 2013-02-06
==================

  * fix package.json
  * fix: Mobile Safari (private mode) is broken with debug
  * fix: Use unicode to send escape character to shell instead of octal to work with strict mode javascript

0.7.1 / 2013-02-05
==================

  * add repository URL to package.json
  * add DEBUG_COLORED to force colored output
  * add browserify support
  * fix component. Closes #24

0.7.0 / 2012-05-04
==================

  * Added .component to package.json
  * Added debug.component.js build

0.6.0 / 2012-03-16
==================

  * Added support for "-" prefix in DEBUG [Vinay Pulim]
  * Added `.enabled` flag to the node version [TooTallNate]

0.5.0 / 2012-02-02
==================

  * Added: humanize diffs. Closes #8
  * Added `debug.disable()` to the CS variant
  * Removed padding. Closes #10
  * Fixed: persist client-side variant again. Closes #9

0.4.0 / 2012-02-01
==================

  * Added browser variant support for older browsers [TooTallNate]
  * Added `debug.enable('project:*')` to browser variant [TooTallNate]
  * Added padding to diff (moved it to the right)

0.3.0 / 2012-01-26
==================

  * Added millisecond diff when isatty, otherwise UTC string

0.2.0 / 2012-01-22
==================

  * Added wildcard support

0.1.0 / 2011-12-02
==================

  * Added: remove colors unless stderr isatty [TooTallNate]

0.0.1 / 2010-01-03
==================

  * Initial release
