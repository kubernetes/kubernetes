0.16.5 / 2015-03-08
-------------------
- fixed `fs.move` when `clobber` is `true` and destination is a directory, it should clobber. https://github.com/jprichardson/node-fs-extra/issues/114

0.16.4 / 2015-03-01
-------------------
- `fs.mkdirs` fix infinite loop on Windows. See: See https://github.com/substack/node-mkdirp/pull/74 and https://github.com/substack/node-mkdirp/issues/66

0.16.3 / 2015-01-28
-------------------
- reverted https://github.com/jprichardson/node-fs-extra/commit/1ee77c8a805eba5b99382a2591ff99667847c9c9


0.16.2 / 2015-01-28
-------------------
- fixed `fs.copy` for Node v0.8 (support is temporary and will be removed in the near future)

0.16.1 / 2015-01-28
-------------------
- if `setImmediate` is not available, fall back to `process.nextTick`

0.16.0 / 2015-01-28
-------------------
- bugfix `fs.move()` into itself. Closes #104
- bugfix `fs.move()` moving directory across device. Closes #108
- added coveralls support
- bugfix: nasty multiple callback `fs.copy()` bug. Closes #98
- misc fs.copy code cleanups

0.15.0 / 2015-01-21
-------------------
- dropped `ncp`, imported code in
- because of previous, now supports `io.js`
- `graceful-fs` is now a dependency

0.14.0 / 2015-01-05
-------------------
- changed `copy`/`copySync` from `fs.copy(src, dest, [filters], callback)` to `fs.copy(src, dest, [options], callback)` https://github.com/jprichardson/node-fs-extra/pull/100
- removed mockfs tests for mkdirp (this may be temporary, but was getting in the way of other tests)

0.13.0 / 2014-12-10
-------------------
- removed `touch` and `touchSync` methods (they didn't handle permissions like UNIX touch)
- updated `"ncp": "^0.6.0"` to `"ncp": "^1.0.1"`
- imported `mkdirp` => `minimist` and `mkdirp` are no longer dependences, should now appease people who wanted `mkdirp` to be `--use_strict` safe. See [#59](https://github.com/jprichardson/node-fs-extra/issues/59)

0.12.0 / 2014-09-22
-------------------
- copy symlinks in `copySync()` [#85](https://github.com/jprichardson/node-fs-extra/pull/85)

0.11.1 / 2014-09-02
-------------------
- bugfix `copySync()` preserve file permissions [#80](https://github.com/jprichardson/node-fs-extra/pull/80)

0.11.0 / 2014-08-11
-------------------
- upgraded `"ncp": "^0.5.1"` to `"ncp": "^0.6.0"`
- upgrade `jsonfile": "^1.2.0"` to `jsonfile": "^2.0.0"` => on write, json files now have `\n` at end. Also adds `options.throws` to `readJsonSync()`
see https://github.com/jprichardson/node-jsonfile#readfilesyncfilename-options for more details.

0.10.0 / 2014-06-29
------------------
* bugfix: upgaded `"jsonfile": "~1.1.0"` to `"jsonfile": "^1.2.0"`, bumped minor because of `jsonfile` dep change
from `~` to `^`. #67

0.9.1 / 2014-05-22
------------------
* removed Node.js `0.8.x` support, `0.9.0` was published moments ago and should have been done there

0.9.0 / 2014-05-22
------------------
* upgraded `ncp` from `~0.4.2` to `^0.5.1`, #58
* upgraded `rimraf` from `~2.2.6` to `^2.2.8`
* upgraded `mkdirp` from `0.3.x` to `^0.5.0`
* added methods `ensureFile()`, `ensureFileSync()`
* added methods `ensureDir()`, `ensureDirSync()` #31
* added `move()` method. From: https://github.com/andrewrk/node-mv


0.8.1 / 2013-10-24
------------------
* copy failed to return an error to the callback if a file doesn't exist (ulikoehler #38, #39)

0.8.0 / 2013-10-14
------------------
* `filter` implemented on `copy()` and `copySync()`. (Srirangan / #36)

0.7.1 / 2013-10-12
------------------
* `copySync()` implemented (Srirangan / #33)
* updated to the latest `jsonfile` version `1.1.0` which gives `options` params for the JSON methods. Closes #32

0.7.0 / 2013-10-07
------------------
* update readme conventions
* `copy()` now works if destination directory does not exist. Closes #29

0.6.4 / 2013-09-05
------------------
* changed `homepage` field in package.json to remove NPM warning

0.6.3 / 2013-06-28
------------------
* changed JSON spacing default from `4` to `2` to follow Node conventions
* updated `jsonfile` dep
* updated `rimraf` dep

0.6.2 / 2013-06-28
------------------
* added .npmignore, #25

0.6.1 / 2013-05-14
------------------
* modified for `strict` mode, closes #24
* added `outputJson()/outputJsonSync()`, closes #23

0.6.0 / 2013-03-18
------------------
* removed node 0.6 support
* added node 0.10 support
* upgraded to latest `ncp` and `rimraf`.
* optional `graceful-fs` support. Closes #17


0.5.0 / 2013-02-03
------------------
* Removed `readTextFile`.
* Renamed `readJSONFile` to `readJSON` and `readJson`, same with write.
* Restructured documentation a bit. Added roadmap.

0.4.0 / 2013-01-28
------------------
* Set default spaces in `jsonfile` from 4 to 2.
* Updated `testutil` deps for tests.
* Renamed `touch()` to `createFile()`
* Added `outputFile()` and `outputFileSync()`
* Changed creation of testing diretories so the /tmp dir is not littered.
* Added `readTextFile()` and `readTextFileSync()`.

0.3.2 / 2012-11-01
------------------
* Added `touch()` and `touchSync()` methods.

0.3.1 / 2012-10-11
------------------
* Fixed some stray globals.

0.3.0 / 2012-10-09
------------------
* Removed all CoffeeScript from tests.
* Renamed `mkdir` to `mkdirs`/`mkdirp`.

0.2.1 / 2012-09-11
------------------
* Updated `rimraf` dep.

0.2.0 / 2012-09-10
------------------
* Rewrote module into JavaScript. (Must still rewrite tests into JavaScript)
* Added all methods of [jsonfile][https://github.com/jprichardson/node-jsonfile]
* Added Travis-CI.

0.1.3 / 2012-08-13
------------------
* Added method `readJSONFile`.

0.1.2 / 2012-06-15
------------------
* Bug fix: `deleteSync()` didn't exist.
* Verified Node v0.8 compatibility.

0.1.1 / 2012-06-15
------------------
* Fixed bug in `remove()`/`delete()` that wouldn't execute the function if a callback wasn't passed.

0.1.0 / 2012-05-31
------------------
* Renamed `copyFile()` to `copy()`. `copy()` can now copy directories (recursively) too.
* Renamed `rmrf()` to `remove()`.
* `remove()` aliased with `delete()`.
* Added `mkdirp` capabilities. Named: `mkdir()`. Hides Node.js native `mkdir()`.
* Instead of exporting the native `fs` module with new functions, I now copy over the native methods to a new object and export that instead.

0.0.4 / 2012-03-14
------------------
* Removed CoffeeScript dependency

0.0.3 / 2012-01-11
------------------
* Added methods rmrf and rmrfSync
* Moved tests from Jasmine to Mocha
