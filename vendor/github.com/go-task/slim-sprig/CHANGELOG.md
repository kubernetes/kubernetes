# Changelog

## Release 3.2.0 (2020-12-14)

### Added

- #211: Added randInt function (thanks @kochurovro)
- #223: Added fromJson and mustFromJson functions (thanks @mholt)
- #242: Added a bcrypt function (thanks @robbiet480)
- #253: Added randBytes function (thanks @MikaelSmith)
- #254: Added dig function for dicts (thanks @nyarly)
- #257: Added regexQuoteMeta for quoting regex metadata (thanks @rheaton)
- #261: Added filepath functions osBase, osDir, osExt, osClean, osIsAbs (thanks @zugl)
- #268: Added and and all functions for testing conditions (thanks @phuslu)
- #181: Added float64 arithmetic addf, add1f, subf, divf, mulf, maxf, and minf
  (thanks @andrewmostello)
- #265: Added chunk function to split array into smaller arrays (thanks @karelbilek)
- #270: Extend certificate functions to handle non-RSA keys + add support for
  ed25519 keys (thanks @misberner)

### Changed

- Removed testing and support for Go 1.12. ed25519 support requires Go 1.13 or newer
- Using semver 3.1.1 and mergo 0.3.11

### Fixed

- #249: Fix htmlDateInZone example (thanks @spawnia)

NOTE: The dependency github.com/imdario/mergo reverted the breaking change in
0.3.9 via 0.3.10 release.

## Release 3.1.0 (2020-04-16)

NOTE: The dependency github.com/imdario/mergo made a behavior change in 0.3.9
that impacts sprig functionality. Do not use sprig with a version newer than 0.3.8.

### Added

- #225: Added support for generating htpasswd hash (thanks @rustycl0ck)
- #224: Added duration filter (thanks @frebib)
- #205: Added `seq` function (thanks @thadc23)

### Changed

- #203: Unlambda functions with correct signature (thanks @muesli)
- #236: Updated the license formatting for GitHub display purposes
- #238: Updated package dependency versions. Note, mergo not updated to 0.3.9
        as it causes a breaking change for sprig. That issue is tracked at
        https://github.com/imdario/mergo/issues/139

### Fixed

- #229: Fix `seq` example in docs (thanks @kalmant)

## Release 3.0.2 (2019-12-13)

### Fixed

- #220: Updating to semver v3.0.3 to fix issue with <= ranges
- #218: fix typo elyptical->elliptic in ecdsa key description (thanks @laverya)

## Release 3.0.1 (2019-12-08)

### Fixed

- #212: Updated semver fixing broken constraint checking with ^0.0

## Release 3.0.0 (2019-10-02)

### Added

- #187: Added durationRound function (thanks @yjp20)
- #189: Added numerous template functions that return errors rather than panic (thanks @nrvnrvn)
- #193: Added toRawJson support (thanks @Dean-Coakley)
- #197: Added get support to dicts (thanks @Dean-Coakley)

### Changed

- #186: Moving dependency management to Go modules
- #186: Updated semver to v3. This has changes in the way ^ is handled
- #194: Updated documentation on merging and how it copies. Added example using deepCopy
- #196: trunc now supports negative values (thanks @Dean-Coakley)

## Release 2.22.0 (2019-10-02)

### Added

- #173: Added getHostByName function to resolve dns names to ips (thanks @fcgravalos)
- #195: Added deepCopy function for use with dicts

### Changed

- Updated merge and mergeOverwrite documentation to explain copying and how to
  use deepCopy with it

## Release 2.21.0 (2019-09-18)

### Added

- #122: Added encryptAES/decryptAES functions (thanks @n0madic)
- #128: Added toDecimal support (thanks @Dean-Coakley)
- #169: Added list contcat (thanks @astorath)
- #174: Added deepEqual function (thanks @bonifaido)
- #170: Added url parse and join functions (thanks @astorath)

### Changed

- #171: Updated glide config for Google UUID to v1 and to add ranges to semver and testify

### Fixed

- #172: Fix semver wildcard example (thanks @piepmatz)
- #175: Fix dateInZone doc example (thanks @s3than)

## Release 2.20.0 (2019-06-18)

### Added

- #164: Adding function to get unix epoch for a time (@mattfarina)
- #166: Adding tests for date_in_zone (@mattfarina)

### Changed

- #144: Fix function comments based on best practices from Effective Go (@CodeLingoTeam)
- #150: Handles pointer type for time.Time in "htmlDate" (@mapreal19)
- #161, #157, #160,  #153, #158, #156,  #155,  #159, #152 documentation updates (@badeadan)

### Fixed

## Release 2.19.0 (2019-03-02)

IMPORTANT: This release reverts a change from 2.18.0

In the previous release (2.18), we prematurely merged a partial change to the crypto functions that led to creating two sets of crypto functions (I blame @technosophos -- since that's me). This release rolls back that change, and does what was originally intended: It alters the existing crypto functions to use secure random.

We debated whether this classifies as a change worthy of major revision, but given the proximity to the last release, we have decided that treating 2.18 as a faulty release is the correct course of action. We apologize for any inconvenience.

### Changed

- Fix substr panic 35fb796 (Alexey igrychev)
- Remove extra period 1eb7729 (Matthew Lorimor)
- Make random string functions use crypto by default 6ceff26 (Matthew Lorimor)
- README edits/fixes/suggestions 08fe136 (Lauri Apple)


## Release 2.18.0 (2019-02-12)

### Added

- Added mergeOverwrite function
- cryptographic functions that use secure random (see fe1de12)

### Changed

- Improve documentation of regexMatch function, resolves #139 90b89ce (Jan Tagscherer)
- Handle has for nil list 9c10885 (Daniel Cohen)
- Document behaviour of mergeOverwrite fe0dbe9 (Lukas Rieder)
- doc: adds missing documentation. 4b871e6 (Fernandez Ludovic)
- Replace outdated goutils imports 01893d2 (Matthew Lorimor)
- Surface crypto secure random strings from goutils fe1de12 (Matthew Lorimor)
- Handle untyped nil values as paramters to string functions 2b2ec8f (Morten Torkildsen)

### Fixed

- Fix dict merge issue and provide mergeOverwrite .dst .src1 to overwrite from src -> dst 4c59c12 (Lukas Rieder)
- Fix substr var names and comments d581f80 (Dean Coakley)
- Fix substr documentation 2737203 (Dean Coakley)

## Release 2.17.1 (2019-01-03)

### Fixed

The 2.17.0 release did not have a version pinned for xstrings, which caused compilation failures when xstrings < 1.2 was used. This adds the correct version string to glide.yaml.

## Release 2.17.0 (2019-01-03)

### Added

- adds alder32sum function and test 6908fc2 (marshallford)
- Added kebabcase function ca331a1 (Ilyes512)

### Changed

- Update goutils to 1.1.0 4e1125d (Matt Butcher)

### Fixed

- Fix 'has' documentation e3f2a85 (dean-coakley)
- docs(dict): fix typo in pick example dc424f9 (Dustin Specker)
- fixes spelling errors... not sure how that happened 4cf188a (marshallford)

## Release 2.16.0 (2018-08-13)

### Added

- add splitn function fccb0b0 (Helgi Þorbjörnsson)
- Add slice func df28ca7 (gongdo)
- Generate serial number a3bdffd (Cody Coons)
- Extract values of dict with values function df39312 (Lawrence Jones)

### Changed

- Modify panic message for list.slice ae38335 (gongdo)
- Minor improvement in code quality - Removed an unreachable piece of code at defaults.go#L26:6 - Resolve formatting issues. 5834241 (Abhishek Kashyap)
- Remove duplicated documentation 1d97af1 (Matthew Fisher)
- Test on go 1.11 49df809 (Helgi Þormar Þorbjörnsson)

### Fixed

- Fix file permissions c5f40b5 (gongdo)
- Fix example for buildCustomCert 7779e0d (Tin Lam)

## Release 2.15.0 (2018-04-02)

### Added

- #68 and #69: Add json helpers to docs (thanks @arunvelsriram)
- #66: Add ternary function (thanks @binoculars)
- #67: Allow keys function to take multiple dicts (thanks @binoculars)
- #89: Added sha1sum to crypto function (thanks @benkeil)
- #81: Allow customizing Root CA that used by genSignedCert (thanks @chenzhiwei)
- #92: Add travis testing for go 1.10
- #93: Adding appveyor config for windows testing

### Changed

- #90: Updating to more recent dependencies
- #73: replace satori/go.uuid with google/uuid (thanks @petterw)

### Fixed

- #76: Fixed documentation typos (thanks @Thiht)
- Fixed rounding issue on the `ago` function. Note, the removes support for Go 1.8 and older

## Release 2.14.1 (2017-12-01)

### Fixed

- #60: Fix typo in function name documentation (thanks @neil-ca-moore)
- #61: Removing line with {{ due to blocking github pages genertion
- #64: Update the list functions to handle int, string, and other slices for compatibility

## Release 2.14.0 (2017-10-06)

This new version of Sprig adds a set of functions for generating and working with SSL certificates.

- `genCA` generates an SSL Certificate Authority
- `genSelfSignedCert` generates an SSL self-signed certificate
- `genSignedCert` generates an SSL certificate and key based on a given CA

## Release 2.13.0 (2017-09-18)

This release adds new functions, including:

- `regexMatch`, `regexFindAll`, `regexFind`, `regexReplaceAll`, `regexReplaceAllLiteral`, and `regexSplit` to work with regular expressions
- `floor`, `ceil`, and `round` math functions
- `toDate` converts a string to a date
- `nindent` is just like `indent` but also prepends a new line
- `ago` returns the time from `time.Now`

### Added

- #40: Added basic regex functionality (thanks @alanquillin)
- #41: Added ceil floor and round functions (thanks @alanquillin)
- #48: Added toDate function (thanks @andreynering)
- #50: Added nindent function (thanks @binoculars)
- #46: Added ago function (thanks @slayer)

### Changed

- #51: Updated godocs to include new string functions (thanks @curtisallen)
- #49: Added ability to merge multiple dicts (thanks @binoculars)

## Release 2.12.0 (2017-05-17)

- `snakecase`, `camelcase`, and `shuffle` are three new string functions
- `fail` allows you to bail out of a template render when conditions are not met

## Release 2.11.0 (2017-05-02)

- Added `toJson` and `toPrettyJson`
- Added `merge`
- Refactored documentation

## Release 2.10.0 (2017-03-15)

- Added `semver` and `semverCompare` for Semantic Versions
- `list` replaces `tuple`
- Fixed issue with `join`
- Added `first`, `last`, `intial`, `rest`, `prepend`, `append`, `toString`, `toStrings`, `sortAlpha`, `reverse`, `coalesce`, `pluck`, `pick`, `compact`, `keys`, `omit`, `uniq`, `has`, `without`

## Release 2.9.0 (2017-02-23)

- Added `splitList` to split a list
- Added crypto functions of `genPrivateKey` and `derivePassword`

## Release 2.8.0 (2016-12-21)

- Added access to several path functions (`base`, `dir`, `clean`, `ext`, and `abs`)
- Added functions for _mutating_ dictionaries (`set`, `unset`, `hasKey`)

## Release 2.7.0 (2016-12-01)

- Added `sha256sum` to generate a hash of an input
- Added functions to convert a numeric or string to `int`, `int64`, `float64`

## Release 2.6.0 (2016-10-03)

- Added a `uuidv4` template function for generating UUIDs inside of a template.

## Release 2.5.0 (2016-08-19)

- New `trimSuffix`, `trimPrefix`, `hasSuffix`, and `hasPrefix` functions
- New aliases have been added for a few functions that didn't follow the naming conventions (`trimAll` and `abbrevBoth`)
- `trimall` and `abbrevboth` (notice the case) are deprecated and will be removed in 3.0.0

## Release 2.4.0 (2016-08-16)

- Adds two functions: `until` and `untilStep`

## Release 2.3.0 (2016-06-21)

- cat: Concatenate strings with whitespace separators.
- replace: Replace parts of a string: `replace " " "-" "Me First"` renders "Me-First"
- plural: Format plurals: `len "foo" | plural "one foo" "many foos"` renders "many foos"
- indent: Indent blocks of text in a way that is sensitive to "\n" characters.

## Release 2.2.0 (2016-04-21)

- Added a `genPrivateKey` function (Thanks @bacongobbler)

## Release 2.1.0 (2016-03-30)

- `default` now prints the default value when it does not receive a value down the pipeline. It is much safer now to do `{{.Foo | default "bar"}}`.
- Added accessors for "hermetic" functions. These return only functions that, when given the same input, produce the same output.

## Release 2.0.0 (2016-03-29)

Because we switched from `int` to `int64` as the return value for all integer math functions, the library's major version number has been incremented.

- `min` complements `max` (formerly `biggest`)
- `empty` indicates that a value is the empty value for its type
- `tuple` creates a tuple inside of a template: `{{$t := tuple "a", "b" "c"}}`
- `dict` creates a dictionary inside of a template `{{$d := dict "key1" "val1" "key2" "val2"}}` 
- Date formatters have been added for HTML dates (as used in `date` input fields)
- Integer math functions can convert from a number of types, including `string` (via `strconv.ParseInt`).

## Release 1.2.0 (2016-02-01)

- Added quote and squote
- Added b32enc and b32dec
- add now takes varargs
- biggest now takes varargs

## Release 1.1.0 (2015-12-29)

- Added #4: Added contains function. strings.Contains, but with the arguments
  switched to simplify common pipelines. (thanks krancour)
- Added Travis-CI testing support

## Release 1.0.0 (2015-12-23)

- Initial release
