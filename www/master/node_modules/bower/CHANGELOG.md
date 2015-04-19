# Changelog

## 1.4.1 - 2015-04-01

- [fix] Reading .bowerrc upwards directory tree ([#1763](https://github.com/bower/bower/issues/1763))
- [fix] Update bower-registry-client so it uses the same bower-config as bower

## 1.4.0 - 2015-03-30

- Add login and unregister commands ([#1719](https://github.com/bower/bower/issues/1719))
- Automatically detecting smart Git hosts ([#1628](https://github.com/bower/bower/issues/1628))
- [bower/config#23] Allow npm config variables ([#1711](https://github.com/bower/bower/issues/1711))
- [bower/config#24] Merge .bowerrc files upwards directory tree ([#1689](https://github.com/bower/bower/issues/1689))
- Better homedir detection (514eb8f)
- Add --save-exact flag ([#1654](https://github.com/bower/bower/issues/1654))
- Ensure extracted files are readable (tar-fs) ([#1548](https://github.com/bower/bower/issues/1548))
- The version command in the programmatic API now returns the new version ([#1755](https://github.com/bower/bower/issues/1755))
- Some minor fixes: #1639, #1620, #1576, #1557, 962a565, a464f5a
- Improved Windows support (AppVeyor CI, tests actually passing on Windows)
- OSX testing enabled on TravisCI

It also includes improved test coverage (~60% -> ~85%) and many refactors.

## 1.3.12 - 2014-09-28

- [stability] Fix versions for unstable dependencies ([#1532](https://github.com/bower/bower/pull/1532))
- [fix] Update tar-fs to support old tar format ([#1537](https://github.com/bower/bower/issues/1537))
- [fix] Make analytics work again ([#1529](https://github.com/bower/bower/pull/1529))
- [fix] Always disable analytics for non-interactive mode ([#1529](https://github.com/bower/bower/pull/1529))
- [fix] Bower init can create private packages again ([#1522](https://github.com/bower/bower/issues/1522))
- [fix] Show again missing newline for bower search output ([#1538](https://github.com/bower/bower/issues/1538))

## 1.3.11 - 2014-09-17

- [fix] Restore install missing dependencies on update ([1519](https://github.com/bower/bower/pull/1519))

## 1.3.10 - 2014-09-13

- [fix] Back down concurrency from 50 to 5 ([#1483](https://github.com/bower/bower/pull/1483))
- [fix] Read .bowerrc from specified cwd ([#1301](https://github.com/bower/bower/pull/1301))
- [fix] Disable shallow clones except those from GitHub ([#1393](https://github.com/bower/bower/pull/1393))
- [fix] Expose bower version ([#1478](https://github.com/bower/bower/pull/1478))
- [fix] Bump dependencies, including "request" ([#1467](https://github.com/bower/bower/pull/1467))
- [fix] Prevent an error when piping bower output to head ([#1508](https://github.com/bower/bower/pull/1508))
- [fix] Disable removing unnecessary resolutions ([#1061](https://github.com/bower/bower/pull/1061))
- [fix] Display the output of hooks again ([#1484](https://github.com/bower/bower/issues/1484))
- [fix] analytics: true in .bowerrc prevents user prompt ([#1470](https://github.com/bower/bower/pull/1470))
- [perf] Use `tar-fs` instead of `tar` for faster TAR extraction ([#1490](https://github.com/bower/bower/pull/1490))

## 1.3.9 - 2014-08-06

- [fix] Handle `tmp` sometimes returning an array ([#1434](https://github.com/bower/bower/pull/1434))

## 1.3.8 - 2014-7-11

- [fix] Lock down `tmp` package dep ([#1403](https://github.com/bower/bower/pull/1403), [#1407](https://github.com/bower/bower/pull/1407))

## 1.3.7 - 2014-07-04

- [fix] callstack error when processing installed packages with circular dependencies ([#1349](https://github.com/bower/bower/issues/1349))
- [fix] Prevent bower list --paths` failing with TypeError ([#1383](https://github.com/bower/bower/issues/1383))
- "bower install" fails if there's no bower.json in current directory ([#922](https://github.com/bower/bower/issues/922))

## 1.3.6 - 2014-07-02

- [fix] Make --force always re-run installation ([#931](https://github.com/bower/bower/issues/931))
- [fix] Disable caching for local resources ([#1356](https://github.com/bower/bower/issues/1356))
- [fix] Emit errors instead throwing them when using bower.commands API ([#1297](https://github.com/bower/bower/issues/1297))
- [fix] Main files and bower.json are never ignored ([#547](https://github.com/bower/bower/issues/547))
- [fix] Check if pkgMeta is undefined during uninstall command ([#1329](https://github.com/bower/bower/issues/1329))
- [fix] Make custom tmp dir and ignores play well with each other ([#1299](https://github.com/bower/bower/issues/1299))
- Warn users when installing package with missing properties ([#694](https://github.com/bower/bower/issues/694))


## 1.3.5 - 2014-06-06
- Search compatible versions in fetching packages ([#1147](https://github.com/bower/bower/issues/1147))

## 1.3.4 - 2014-06-02

- Resolve a situation in which the install process gets into an infinite loop ([#1169](https://github.com/bower/bower/issues/1169))
- Improved CLI output for conflicts ([#1284](https://github.com/bower/bower/issues/1284))
- Changed `bower version` to mirror the tag format of `npm version` ([#1278](https://github.com/bower/bower/issues/1278))
- Allow short commit SHAs to be used ([#990](https://github.com/bower/bower/issues/990))

## 1.3.3 - 2014-04-24

- Do not cache moving targets like branches ([#1242](https://github.com/bower/bower/issues/1242))
- Suppress output if --quiet option is specified ([#1124](https://github.com/bower/bower/pull/1124))
- Use "svn export" for efficiency ([#1224](https://github.com/bower/bower/pull/1224))
- Prevent loading insights and analytics on CI ([#1221](https://github.com/bower/bower/issues/1221))
- Make "bower list" respect custom components directory ([#1237](https://github.com/bower/bower/issues/1237))
- Improve non-interactive loading performance 2x ([#1238](https://github.com/bower/bower/issues/1238))
- Load commands only on demand, improving performance ([#1232](https://github.com/bower/bower/pull/1232))

## 1.3.2 - 2014-04-05

- Added yui moduleType [PR #1129](https://github.com/bower/bower/pull/1129)
- Fixes for concurrency issues [PR #1211](https://github.com/bower/bower/pull/1211)
- `link` now installs package dependencies [PR #891](https://github.com/bower/bower/pull/891)
- Improved conflict installation message [Commit](https://github.com/bower/bower/commit/bea533acf87903d4b411bfbaa7df93f852ef46a3)
- Add --production switch to "prune" command [PR #1168](https://github.com/bower/bower/pull/1168)


## 1.3.1 - 2014-03-10

- No longer ask for permission to gather analytics when running on in a CI environment.


## 1.3.0 - 2014-03-10

- **Removed support for node 0.8.** It may still work but we will no longer fix bugs for older versions of node.
- Add **Bower Insight** for opt-in analytics integration to help improve tool and gain insight on community trends
  - Old overview of [Insight](https://github.com/yeoman/yeoman/wiki/Insight), [Issue #260](https://github.com/bower/bower/issues/260)
  - Reporting to GA. Public Dashboard is in progress.
  - [Turn off interactive mode](https://github.com/bower/bower/issues/1162) if you run Bower in a CI environment
- Add `moduleType` property to bower init ([#934](https://github.com/bower/bower/pull/934))
- Fix prune command to log only after cleanup is completed ([#1023](https://github.com/bower/bower/issues/1023))
- Fix git resolver to ignore pre-release versions ([#1017](https://github.com/bower/bower/issues/1017))
- Fix shorthand flag for `save` option on `uninstall` command ([#1031](https://github.com/bower/bower/pull/1031))
- Add `bower version` command ([#961](https://github.com/bower/bower/pull/961))
- Add .bowerrc option to use `--save` by default when using `bower install` command ([#1074](https://github.com/bower/bower/pull/1074))
- Fix git resolver caching ([#1083](https://github.com/bower/bower/issues/1083))
- Fix reading versions from cache directory ([#1076](https://github.com/bower/bower/pull/1076))
- Add svn support ([#1055](https://github.com/bower/bower/pull/1055))
- Allow circular dependencies to be installed ([#1104](https://github.com/bower/bower/pull/1104))
- Add scripts/hooks support ([#718](https://github.com/bower/bower/pull/718))

_NOTE_: It's advisable that users use `--config.interactive=false` on automated scripts.


## 1.2.8 - 2013-12-02
- Fix absolute paths ending with / not going through the FsResolver, ([#898](https://github.com/bower/bower/issues/898))
- Allow query string parameters in package URLs
- Swapped 'unzip' module for 'decompress-zip', and some other small unzipping fixes([#873](https://github.com/bower/bower/issues/873), [#896](https://github.com/bower/bower/issues/896))
- Allow the root-check to be overridden when calling bower programmatically.
- Fixed some bugs relating to packages with a very large dependency tree
- Fix a bug caused by a recent change to semver


## 1.2.7 - 2013-09-29

- Do not swallow sync errors when using the programmatic API ([#849](https://github.com/bower/bower/issues/849))
- Fix resolutions not being saved if `--force-latest` is specified ([#861](https://github.com/bower/bower/issues/861))
- Fix `bower register` warning about URL conversion, even if no conversion occurred
- Fix `bower update` not correctly catching up branch commits
- Add configured directory in `.bowerrc` to the ignores in `bower init` ([#854](https://github.com/bower/bower/issues/854))
- Fix some case sensitive issues with data stored in registry cache (e.g.: jquery/jQuery, [#859](https://github.com/bower/bower/issues/859))
- Fix bower not checking out a tag if it looks like a semver (e.g.: 1.0, [#872](https://github.com/bower/bower/issues/872))
- Fix install & update commands printing the wrong versions in some cases ([#879](https://github.com/bower/bower/issues/879))
- Give priority to mime type headers when deciding if a package need to be extracted, except if it is `octet-stream`

_NOTE_: It's advisable that users run `bower cache clean`.


## 1.2.6 - 2013-09-04

- Bower now reports download progress even for servers that do not respond with `content-length` header.
- Do not translate endpoints when registering a package to a private registry server ([#832](https://github.com/bower/bower/issues/832))
- Detect corrupted downloads by comparing downloaded bytes with `content-length` header if possible; this fixes Bower silently failing on unstable networks ([#824](https://github.com/bower/bower/issues/824) and [#792](https://github.com/bower/bower/issues/792))
- Fix quotes in fields causing Bower to crash in the `init` command ([#841](https://github.com/bower/bower/issues/841))


## 1.2.5 - 2013-08-28

- Fix persistent conflict resolutions not working correctly for branches ([#818](https://github.com/bower/bower/issues/818))
- Fix Bower failing to run if HOME is not set ([#826](https://github.com/bower/bower/issues/826))
- Bower now prints a warning if HOME is not set ([#827](https://github.com/bower/bower/issues/827))
- Fix progress message being fired after completion of long running `git clone` commands
- Other minor improvements


## 1.2.4 - 2013-08-23

- Fix ignored nested folders not being correctly handled in some cases ([#814](https://github.com/bower/bower/issues/814))


## 1.2.3 - 2013-08-22

- Fix read of environment variables that map to config properties with dashes and also support nested ones ([#8@bower-config](https://github.com/bower/config/issues/8))
- Fix `bower info <package> <property>` printing the available versions (it shouldn't!)
- Fix interactive shell not being correctly detected in node `0.8.x` ([#802](https://github.com/bower/bower/issues/802))
- Fix `extraneous` flag in the `list` command being incorrectly set for saved dev dependencies in some cases
- Fix linked dependencies not being read in `bower list` on Windows ([#813](https://github.com/bower/bower/issues/813))
- Fix update notice not working with `--json`


## 1.2.2 - 2013-08-20

- Standardize prompt behaviour with and without `--json`
- Improve detection of `git` servers that do not support shallow clones ([#805](https://github.com/bower/bower/issues/805))
- Ignore remote tags (tags ending with ^{})
- Fix bower not saving the correct endpoint in some edge cases ([#806](https://github.com/bower/bower/issues/806))


## 1.2.1 - 2013-08-19

- Fix bower throwing on non-semver targets ([#800](https://github.com/bower/bower/issues/800))


## 1.2.0 - 2013-08-19

- __Bower no longer installs a pre-release version by default, that is, if no version/range is specified__ ([#782](https://github.com/bower/bower/issues/782))
- __`bower info <package>` will now show the latest `<package>` information along with the available versions__ ([#759](https://github.com/bower/bower/issues/759))
- __`bower link` no longer requires an elevated user on Windows in most cases__ ([#472](https://github.com/bower/bower/issues/472))
- __Init command now prompts for the whole `bower.json` spec properties, filling in default values for `author` and `homepage` based on `git` settings__ ([#693](https://github.com/bower/bower/issues/693))
- Changes to endpoint sources in `bower.json` are now catched up by `bower install` and `bower update` ([#788](https://github.com/bower/bower/issues/788))
- Allow semver ranges in `bower cache clean`, e.g. `bower cache clean jquery#<2.0.0` ([#688](https://github.com/bower/bower/issues/688))
- Normalize `bower list --paths` on Windows ([#279](https://github.com/bower/bower/issues/279))
- Multiple mains are now correctly outputted as an array in `bower list --paths` ([#784](https://github.com/bower/bower/issues/784))
- Add `--relative` option to `bower list --json` so that Bower outputs relative paths instead of absolute ([#714](https://github.com/bower/bower/issues/714))
- `bower list --paths` now outputs relative paths by default; can be turned off with `--no-relative` ([#785](https://github.com/bower/bower/issues/785))
- Bower no longer fails if `symlinks` to files are present in the `bower_components` folder ([#783](https://github.com/bower/bower/issues/783) and [#791](https://github.com/bower/bower/issues/791))
- Disable git templates/hooks when running `git` ([#761](https://github.com/bower/bower/issues/761))
- Add instructions to setup git workaround for proxies when execution of `git` fails ([#250](https://github.com/bower/bower/issues/250))
- Ignore `component.json` if it looks like a component(1) file ([#556](https://github.com/bower/bower/issues/556))
- Fix multi-user usage on bower when it creates temporary directories to hold some files
- Fix prompting causing an invalid JSON output when running commands with `--json`
- When running Bower commands programmatically, prompting is now disabled by default (see the updated programmatic [usage](https://github.com/bower/bower#programmatic-api) for more info)
- Other minor improvements and fixes

Fix for `#788` requires installed components to be re-installed.


## 1.1.2 - 2013-08-10

- Detect and fallback if the git server does not support `--depth=1` when cloning ([#747](https://github.com/bower/bower/issues/747))


## 1.1.1 - 2013-08-08

- Fix silent fail when spawning child processes in some edge cases ([#722](https://github.com/bower/bower/issues/722))
- Fix `home` command not guessing the correct URL for `GitHub` ssh endpoints (requires `bower cache-clean`)
- Fix bower not correctly filtering packages with symlinks in some cases ([#730](https://github.com/bower/bower/issues/730))
- Fix multi-user usage on bower when it falls back to create a `/tmp/bower` folder ([#743](https://github.com/bower/bower/issues/743))
- Bower now sends a fake user agent when behind a proxy by default, so that corporate proxies do not block requests ([#698](https://github.com/bower/bower/issues/698))
- Bower now translates GitHub public `git://` URLs to `git@` when behind a proxy ([#731](https://github.com/bower/bower/issues/731))
- Minor improvements to the CLI output on small terminals
- Minor programmatic usage improvements
- Minor help usage fixes


## 1.1.0 - 2013-08-03

- __Fix `--save` and `--save-dev` not working correctly for the uninstall command in some situations__
- __Attempting to register a package that declares `"private": true` in `bower.json` will result in an error ([#162](https://github.com/bower/bower/issues/162))__
- __Fix retry strategy on download error that was causing some strange I/O errors__ ([#699](https://github.com/bower/bower/issues/699) and [#704](https://github.com/bower/bower/issues/704))
- __`bower prune` now clears pruned packages dependencies if they are also extraneous__ ([#708](https://github.com/bower/bower/issues/708))
- __`bower uninstall` now uninstalls uninstalled packages dependencies if they are not shared ([#609](https://github.com/bower/bower/issues/609))__
- Fix `bower list` display the `incompatible` label even if they are compatible ([#710](https://github.com/bower/bower/issues/710))
- Fix `bower cache clean` not working correctly when `package#non-semver` is specified
- Implement no operation `completion` command to prevent weird output when hitting tab ([#691](https://github.com/bower/bower/issues/691))
- Fix `bower info --help` ([#703](https://github.com/bower/bower/issues/703))
- Add colorized output for `bower info <package>#<version>` ([#571](https://github.com/bower/bower/issues/571))
- Added `bower ls` as an alias to `bower list`
- Fix regression: do not create a json file when saving is required, warn instead
- Ignore linked packages when reading dependencies in `bower init` ([#709](https://github.com/bower/bower/issues/709))
- `bower list` is now able to (partially) reconstruct the dependency tree, even for dependencies not declared in `bower.json` ([#622](https://github.com/bower/bower/issues/622))


## 1.0.3 - 2013-07-30

- Fix some changes not being saved to bower.json ([#685](https://github.com/bower/bower/issues/685))
- Fix `bower info <package> <property>` not showing information related to property of the latest version of that package ([#684](https://github.com/bower/bower/issues/684))


## 1.0.2 - 2013-07-30

- Fix severe bug originated from a wrong merge that caused conflict messages to not show up correctly


## 1.0.1 - 2013-07-29

- Fix `bower register` going ahead even if the answer was `no` ([#644](https://github.com/bower/bower/issues/644))
- Fix local endpoints with backslashes on Windows ([#2@endpoint-parser](https://github.com/bower/endpoint-parser/pull/2))
- Fix usage of multiple registries in the registry-client ([#3@registry-client](https://github.com/bower/registry-client/pull/3) and [#2@registry-client](https://github.com/bower/registry-client/pull/2))
- File extensions now have more priority than mime types when deciding if extraction is necessary ([#657](https://github.com/bower/bower/pull/657))
- Fix `Bower` not working when calling `.bat`/`.cmd` commands on Windows; it affected people using `Git portable` ([#626](https://github.com/bower/bower/issues/626))
- Fix `bower list --paths` not resolving all files to absolute paths when the `main` property contained multiple files ([660](https://github.com/bower/bower/issues/660))
- Fix `Bower` renaming `bower.json` and `component.json` files to `index.json` when it was the only file in the folder ([#674](https://github.com/bower/bower/issues/674))
- Ignore symlinks when copying/extracting since they are not portable, specially across different hard-drives ([#665](https://github.com/bower/bower/issues/665))
- Local file/dir endpoints are now exclusively referenced by an absolute path or relative path starting with `.` ([#666](https://github.com/bower/bower/issues/666))
- Linked packages `bower.json` files are now parsed, making `bower list` account linked packages dependencies ([#659](https://github.com/bower/bower/issues/659))
- Bower now fails to run with sudo unless `--allow-root` is passed ([#498](https://github.com/bower/bower/issues/498))
- Add additional system information such as node version, bower version, OS version when an error occurs ([#670](https://github.com/bower/bower/issues/670))
- `bower install` no longer overwrites `linked` packages unless it needs to ([#593](https://github.com/bower/bower/issues/593)).
- All endpoint parts are now trimmed so that the Manager can better detect similar endpoints ([#3@endpoint-parser](https://github.com/bower/endpoint-parser/pull/3))
- `bower register` now shows the server that will be used ([#647](https://github.com/bower/endpoint-parser/pull/647))


## 1.0.0 - 2013-07-23

Total rewrite of bower.
The list bellow highlights the most important stuff.
For a complete list of changes that this rewrite and release brings please read: https://github.com/bower/bower/wiki/Rewrite-state


- Clear architecture and separation of concerns
- Much much faster
- `--json` output for all commands
- `--offline` usage for all commands, except `register`
- Proper `install` and `update` commands, similar to `npm` in behaviour
- Named endpoints when installing, e.g. `bower install backbone-amd=backbone#~1.0.0`
- New interactive conflict resolution strategy
- Prevent human errors when using `register`
- New `home` command, similar to `npm`
- New `cache list` command
- New `prune` command
- Many many general bug fixes

Non-backwards compatible changes:

- The value of the `json` property from .bowerrc is no longer used
- `--map` and `--sources` from the list command were removed, use `--json` instead
- Programmatic usage changed, specially the commands interface

Users upgrading from `bower-canary` and `bower@~0.x.x` should do a `bower cache clean`.
Additionally you may remove the `~/.bower` folder manually since it's no longer used.
On Windows the folder is located in `AppData/bower`.


## 0.10.0 - 2013-07-02

- __Allow specific commits to be targeted__ ([#275](https://github.com/bower/bower/issues/275))
- __Change bower default folder from `components` to `bower_components`__ ([#434](https://github.com/bower/bower/issues/434))
- __Support semver pre-releases and builds__ ([#188](https://github.com/bower/bower/issues/188))
- Use `Content-Type` and `Content-Disposition` to guess file types, such as zip files ([#454](https://github.com/bower/bower/pull/454))
- Fix bower failing silently when using an invalid version value in the bower.json file ([#439](https://github.com/bower/bower/issues/439))
- Fix bower slowness when downloading after redirects ([#437](https://github.com/bower/bower/issues/437))
- Detect and error out with a friendly message when `git` is not installed ([#362](https://github.com/bower/bower/issues/362))
- Add `--quiet` and `--silent` CLI options ([#343](https://github.com/bower/bower/issues/343))
- Minor programmatic usage improvements

_NOTE_: The `components` folder will still be used if already created, making it easier for users to upgrade.

## 0.9.2 - 2013-04-28
- Better fix for [#429](https://github.com/bower/bower/issues/429)

## 0.9.1 - 2013-04-27
- Update `package.json`, docs and other stuff to point to the new `Bower` organisation on GitHub
- Fix root label of `bower list` being an absolute path; now uses the package name
- Fix `bower update <pkg>` updating all packages; now throws when updating an unknown package
- Fix `list` command when package use different names than the `guessed` one ([#429](https://github.com/bower/bower/issues/429))

## 0.9.0 - 2013-04-25
- __Change from `component.json` to `bower.json`__ ([#39](https://github.com/bower/bower/issues/39))
- __Compatibility with `node 0.10.x`, including fix hangs/errors when extracting `zip` files__
- Fix `--save` and `--save-dev` not working with URLs that get redirected ([#417](https://github.com/bower/bower/issues/417))
- Fix `init` command targeting `~commit` instead of `*`. ([#385](https://github.com/bower/bower/issues/385))
- Remove temporary directories before exiting ([#345](https://github.com/bower/bower/issues/345))
- Integrate `update-notifier` ([#202](https://github.com/bower/bower/issues/202))
- Use `json` name when a package name was inferred ([#192](https://github.com/bower/bower/issues/192))
- Fix `bin/bower` not exiting with an exit code greater than zero when an error occurs ([#187](https://github.com/bower/bower/issues/187))
- Fix `--save` and `--save-dev` saving resolved shorthands instead of the actual shorthands
- Fix bower using user defined git templates ([#324](https://github.com/bower/bower/issues/324))
- Add command abbreviations ([#262](https://github.com/bower/bower/issues/262))
- Improve help messages and fix abuse of colors in output
- Wait for every package to resolve before printing error messages ([#290](https://github.com/bower/bower/issues/290))
- Add `shorthand_resolver` to allow shorthands to be resolved to repositories other than GitHub ([#278](https://github.com/bower/bower/issues/278))

## 0.8.6 - 2013-04-03
- Emergency fix for `node 0.8.x` users to make `zip` extraction work again

## 0.8.5 - 2013-03-04
- Fix `cache-clean` command clearing the completion cache when the command was called with specific packages
- Add error message when an error is caught parsing an invalid `component.json`

## 0.8.4 - 2013-03-01
- Fix some more duplicate async callbacks being called twice
- Preserve new lines when saving `component.json` ([#285](https://github.com/bower/bower/issues/285))

## 0.8.3 - 2013-02-27
- Fix error when using the `update` command ([#282](https://github.com/bower/bower/issues/282))

## 0.8.2 - 2013-02-26
- Fix some errors in windows while removing directories, had to downgrade `rimraf` ([#274](https://github.com/bower/bower/issues/274))
- Prevent duplicate package names in error summaries ([#277](https://github.com/bower/bower/issues/277))

## 0.8.1 - 2013-02-25
- Fix some async callbacks being fired twice ([#274](https://github.com/bower/bower/issues/274))

## 0.8.0 - 2013-02-24
- __Add init command similar to `npm init`__ ([#219](https://github.com/bower/bower/issues/219))
- __Add devDependencies__ support ([#251](https://github.com/bower/bower/issues/251))
- __Add `--save-dev` flag to install/uninstall commands__ ([#258](https://github.com/bower/bower/issues/258))
- `cache-clean` command now clears links pointing to nonexistent folders ([#182](https://github.com/bower/bower/issues/182))
- Fix issue when downloading assets behind a proxy using `https` ([#230](https://github.com/bower/bower/issues/230))
- Fix --save saving unresolved components ([#240](https://github.com/bower/bower/issues/240))
- Fix issue when extracting some zip files ([#225](https://github.com/bower/bower/issues/225))
- Fix automatic conflict resolver not selecting the correct version
- Add `--sources` option to the `list` command ([#235](https://github.com/bower/bower/issues/235))
- Automatically clear cache when git commands fail with code 128 ([#216](https://github.com/bower/bower/issues/216))
- Fix `bower` not working correctly behind a proxy in some commands ([#208](https://github.com/bower/bower/issues/208))

## 0.7.1 - 2013-02-20
- Remove postinstall script from `bower` installation

## 0.7.0 - 2013-02-01
- __Ability to resolve conflicts__ ([#214](https://github.com/bower/bower/issues/214))
- __Ability to search and publish to different endpoints by specifying them in the `.bowerrc` file__
- __Experimental autocompletion__
- __Ability to exclude (ignore) files__
- Fix minor issues in the cache clean command
- Better error message for invalid semver tags ([#185](https://github.com/bower/bower/issues/185))
- Only show discover message in the list command only if there are packages
- Fix mismatch issue due to reading cached component.json files ([#214](https://github.com/bower/bower/issues/214))
- Better error messages when reading invalid .bowerrc files ([#220](https://github.com/bower/bower/issues/220))
- Fix update command when used in packages pointing to assets ([#197](https://github.com/bower/bower/issues/197))
- Bower now obeys packages's `.bowerrc` if they define a different `json` ([#205](https://github.com/bower/bower/issues/205))

## 0.6.8 - 2012-12-14
- Improve list command
  - Does not fetch versions if not necessary (for --map and --paths options)
  - Add --offline option to prevent versions from being fetched
- Fix uninstall command not firing the `end` event
- Fix error when executing an unknown command ([#179](https://github.com/bower/bower/issues/179))
- Fix help for the ls command (alias of list)

## 0.6.7 - 2012-12-10
- Fix uninstall removing all unsaved dependencies ([#178](https://github.com/bower/bower/issues/178))
- Fix uninstall --force flag in some cases
- Add --silent option to the register option, to avoid questioning
- Fix possible issues with options in some commands
- Fix error reporting when reading invalid project component.json

## 0.6.6 - 2012-12-03
- Improve error handling while reading component.json
- Fix package name not being correctly collected in the error summary

## 0.6.5 - 2012-12-01
- Fix error summary not being displayed in some edge cases
- Fix bower not fetching latest commits correctly in some cases

## 0.6.4 - 2012-11-29
- Fix permission on downloaded files ([#160](https://github.com/bower/bower/issues/160))

## 0.6.3 - 2012-11-24
- Fix version not being correctly set for local packages ([#155](https://github.com/bower/bower/issues/155))

## 0.6.2 - 2012-11-23
- Fix uninstall --save when there is no component.json

## 0.6.1 - 2012-11-22
- Fix uninstall when the project component.json has no deps saved ([#153](https://github.com/bower/bower/issues/153))
- Fix uncaught errors when using file writer (they are now caught and reported)
- Fix temporary directories not being deleted when an exception occurs ([#153](https://github.com/bower/bower/issues/140))

## 0.6.0 - 2012-11-21
- __Add link command__ (similar to npm)
- Fix error reporting for nested deps
- Abort if a repository is detected when installing.
  This is useful to prevent people from loosing their work
- Minor fixes and improvements

## 0.5.1 - 2012-11-20
- Add errors summary to the end of install/update commands
- Add windows instructions to the README

## 0.5.0 - 2012-11-19
- __Remove package.json support__
- __Support for local path repositories__ ([#132](https://github.com/bower/bower/issues/132))
- `install --save` now saves the correct tag (e.g: ~0.0.1) instead of 'latest'
- `install --save` now saves packages pointing directly to assets correctly
- Bower automatically creates a component.json when install with `--save` is used
- Fix issues with list command ([#142](https://github.com/bower/bower/issues/142))
- Fix local paths not being saved when installing with --save ([#114](https://github.com/bower/bower/issues/114))
- `uninstall` now uninstalls nested dependencies if they are not shared ([#83](https://github.com/bower/bower/issues/83))
- `uninstall` now warns when a dependency conflict occurs and aborts.
  It will only proceed if the `--force` flag is passed
- Bower now detects mismatches between the version specified in the component.json and the tag, informing the user
- `bower ls` now informs when a package has a new commit (for non-tagged repos)
- Add jshintrc and fix a lot of issues related with JSHint warnings
- `bower register` now prompts if the user really wants to proceed
