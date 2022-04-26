## Changelog

### [1.8.2](https://github.com/magiconair/properties/tree/v1.8.2) - 25 Aug 2020

 * [PR #36](https://github.com/magiconair/properties/pull/36): Escape backslash on write

   This patch ensures that backslashes are escaped on write. Existing applications which
   rely on the old behavior may need to be updated.

   Thanks to [@apesternikov](https://github.com/apesternikov) for the patch.

 * [PR #42](https://github.com/magiconair/properties/pull/42): Made Content-Type check whitespace agnostic in LoadURL()

   Thanks to [@aliras1](https://github.com/aliras1) for the patch.

 * [PR #41](https://github.com/magiconair/properties/pull/41): Make key/value separator configurable on Write()

   Thanks to [@mkjor](https://github.com/mkjor) for the patch.

 * [PR #40](https://github.com/magiconair/properties/pull/40): Add method to return a sorted list of keys

   Thanks to [@mkjor](https://github.com/mkjor) for the patch.

### [1.8.1](https://github.com/magiconair/properties/tree/v1.8.1) - 10 May 2019

 * [PR #35](https://github.com/magiconair/properties/pull/35): Close body always after request

   This patch ensures that in `LoadURL` the response body is always closed.

   Thanks to [@liubog2008](https://github.com/liubog2008) for the patch.

### [1.8](https://github.com/magiconair/properties/tree/v1.8) - 15 May 2018

 * [PR #26](https://github.com/magiconair/properties/pull/26): Disable expansion during loading

   This adds the option to disable property expansion during loading.

   Thanks to [@kmala](https://github.com/kmala) for the patch.

### [1.7.6](https://github.com/magiconair/properties/tree/v1.7.6) - 14 Feb 2018

 * [PR #29](https://github.com/magiconair/properties/pull/29): Reworked expansion logic to handle more complex cases.

   See PR for an example.

   Thanks to [@yobert](https://github.com/yobert) for the fix.

### [1.7.5](https://github.com/magiconair/properties/tree/v1.7.5) - 13 Feb 2018

 * [PR #28](https://github.com/magiconair/properties/pull/28): Support duplicate expansions in the same value

   Values which expand the same key multiple times (e.g. `key=${a} ${a}`) will no longer fail
   with a `circular reference error`.

   Thanks to [@yobert](https://github.com/yobert) for the fix.

### [1.7.4](https://github.com/magiconair/properties/tree/v1.7.4) - 31 Oct 2017

 * [Issue #23](https://github.com/magiconair/properties/issues/23): Ignore blank lines with whitespaces

 * [PR #24](https://github.com/magiconair/properties/pull/24): Update keys when DisableExpansion is enabled

   Thanks to [@mgurov](https://github.com/mgurov) for the fix.

### [1.7.3](https://github.com/magiconair/properties/tree/v1.7.3) - 10 Jul 2017

 * [Issue #17](https://github.com/magiconair/properties/issues/17): Add [SetValue()](http://godoc.org/github.com/magiconair/properties#Properties.SetValue) method to set values generically
 * [Issue #22](https://github.com/magiconair/properties/issues/22): Add [LoadMap()](http://godoc.org/github.com/magiconair/properties#LoadMap) function to load properties from a string map

### [1.7.2](https://github.com/magiconair/properties/tree/v1.7.2) - 20 Mar 2017

 * [Issue #15](https://github.com/magiconair/properties/issues/15): Drop gocheck dependency
 * [PR #21](https://github.com/magiconair/properties/pull/21): Add [Map()](http://godoc.org/github.com/magiconair/properties#Properties.Map) and [FilterFunc()](http://godoc.org/github.com/magiconair/properties#Properties.FilterFunc)

### [1.7.1](https://github.com/magiconair/properties/tree/v1.7.1) - 13 Jan 2017

 * [Issue #14](https://github.com/magiconair/properties/issues/14): Decouple TestLoadExpandedFile from `$USER`
 * [PR #12](https://github.com/magiconair/properties/pull/12): Load from files and URLs
 * [PR #16](https://github.com/magiconair/properties/pull/16): Keep gofmt happy
 * [PR #18](https://github.com/magiconair/properties/pull/18): Fix Delete() function

### [1.7.0](https://github.com/magiconair/properties/tree/v1.7.0) - 20 Mar 2016

 * [Issue #10](https://github.com/magiconair/properties/issues/10): Add [LoadURL,LoadURLs,MustLoadURL,MustLoadURLs](http://godoc.org/github.com/magiconair/properties#LoadURL) method to load properties from a URL.
 * [Issue #11](https://github.com/magiconair/properties/issues/11): Add [LoadString,MustLoadString](http://godoc.org/github.com/magiconair/properties#LoadString) method to load properties from an UTF8 string.
 * [PR #8](https://github.com/magiconair/properties/pull/8): Add [MustFlag](http://godoc.org/github.com/magiconair/properties#Properties.MustFlag) method to provide overrides via command line flags. (@pascaldekloe)

### [1.6.0](https://github.com/magiconair/properties/tree/v1.6.0) - 11 Dec 2015

 * Add [Decode](http://godoc.org/github.com/magiconair/properties#Properties.Decode) method to populate struct from properties via tags.

### [1.5.6](https://github.com/magiconair/properties/tree/v1.5.6) - 18 Oct 2015

 * Vendored in gopkg.in/check.v1

### [1.5.5](https://github.com/magiconair/properties/tree/v1.5.5) - 31 Jul 2015

 * [PR #6](https://github.com/magiconair/properties/pull/6): Add [Delete](http://godoc.org/github.com/magiconair/properties#Properties.Delete) method to remove keys including comments. (@gerbenjacobs)

### [1.5.4](https://github.com/magiconair/properties/tree/v1.5.4) - 23 Jun 2015

 * [Issue #5](https://github.com/magiconair/properties/issues/5): Allow disabling of property expansion [DisableExpansion](http://godoc.org/github.com/magiconair/properties#Properties.DisableExpansion). When property expansion is disabled Properties become a simple key/value store and don't check for circular references.

### [1.5.3](https://github.com/magiconair/properties/tree/v1.5.3) - 02 Jun 2015

 * [Issue #4](https://github.com/magiconair/properties/issues/4): Maintain key order in [Filter()](http://godoc.org/github.com/magiconair/properties#Properties.Filter), [FilterPrefix()](http://godoc.org/github.com/magiconair/properties#Properties.FilterPrefix) and [FilterRegexp()](http://godoc.org/github.com/magiconair/properties#Properties.FilterRegexp)

### [1.5.2](https://github.com/magiconair/properties/tree/v1.5.2) - 10 Apr 2015

 * [Issue #3](https://github.com/magiconair/properties/issues/3): Don't print comments in [WriteComment()](http://godoc.org/github.com/magiconair/properties#Properties.WriteComment) if they are all empty
 * Add clickable links to README

### [1.5.1](https://github.com/magiconair/properties/tree/v1.5.1) - 08 Dec 2014

 * Added [GetParsedDuration()](http://godoc.org/github.com/magiconair/properties#Properties.GetParsedDuration) and [MustGetParsedDuration()](http://godoc.org/github.com/magiconair/properties#Properties.MustGetParsedDuration) for values specified compatible with
   [time.ParseDuration()](http://golang.org/pkg/time/#ParseDuration).

### [1.5.0](https://github.com/magiconair/properties/tree/v1.5.0) - 18 Nov 2014

 * Added support for single and multi-line comments (reading, writing and updating)
 * The order of keys is now preserved
 * Calling [Set()](http://godoc.org/github.com/magiconair/properties#Properties.Set) with an empty key now silently ignores the call and does not create a new entry
 * Added a [MustSet()](http://godoc.org/github.com/magiconair/properties#Properties.MustSet) method
 * Migrated test library from launchpad.net/gocheck to [gopkg.in/check.v1](http://gopkg.in/check.v1)

### [1.4.2](https://github.com/magiconair/properties/tree/v1.4.2) - 15 Nov 2014

 * [Issue #2](https://github.com/magiconair/properties/issues/2): Fixed goroutine leak in parser which created two lexers but cleaned up only one

### [1.4.1](https://github.com/magiconair/properties/tree/v1.4.1) - 13 Nov 2014

 * [Issue #1](https://github.com/magiconair/properties/issues/1): Fixed bug in Keys() method which returned an empty string

### [1.4.0](https://github.com/magiconair/properties/tree/v1.4.0) - 23 Sep 2014

 * Added [Keys()](http://godoc.org/github.com/magiconair/properties#Properties.Keys) to get the keys
 * Added [Filter()](http://godoc.org/github.com/magiconair/properties#Properties.Filter), [FilterRegexp()](http://godoc.org/github.com/magiconair/properties#Properties.FilterRegexp) and [FilterPrefix()](http://godoc.org/github.com/magiconair/properties#Properties.FilterPrefix) to get a subset of the properties

### [1.3.0](https://github.com/magiconair/properties/tree/v1.3.0) - 18 Mar 2014

* Added support for time.Duration
* Made MustXXX() failure beha[ior configurable (log.Fatal, panic](https://github.com/magiconair/properties/tree/vior configurable (log.Fatal, panic) - custom)
* Changed default of MustXXX() failure from panic to log.Fatal

### [1.2.0](https://github.com/magiconair/properties/tree/v1.2.0) - 05 Mar 2014

* Added MustGet... functions
* Added support for int and uint with range checks on 32 bit platforms

### [1.1.0](https://github.com/magiconair/properties/tree/v1.1.0) - 20 Jan 2014

* Renamed from goproperties to properties
* Added support for expansion of environment vars in
  filenames and value expressions
* Fixed bug where value expressions were not at the
  start of the string

### [1.0.0](https://github.com/magiconair/properties/tree/v1.0.0) - 7 Jan 2014

* Initial release
