## Changelog

### [1.7.0](https://github.com/magiconair/properties/tags/v1.7.0) - 20 Mar 2016

 * [Issue #10](https://github.com/magiconair/properties/issues/10): Add [LoadURL,LoadURLs,MustLoadURL,MustLoadURLs](http://godoc.org/github.com/magiconair/properties#Properties.LoadURL) method to load properties from a URL.
 * [Issue #11](https://github.com/magiconair/properties/issues/11): Add [LoadString,MustLoadString](http://godoc.org/github.com/magiconair/properties#Properties.LoadString) method to load properties from an UTF8 string.
 * [PR #8](https://github.com/magiconair/properties/pull/8): Add [MustFlag](http://godoc.org/github.com/magiconair/properties#Properties.MustFlag) method to provide overrides via command line flags. (@pascaldekloe)

### [1.6.0](https://github.com/magiconair/properties/tags/v1.6.0) - 11 Dec 2015

 * Add [Decode](http://godoc.org/github.com/magiconair/properties#Properties.Decode) method to populate struct from properties via tags.

### [1.5.6](https://github.com/magiconair/properties/tags/v1.5.6) - 18 Oct 2015

 * Vendored in gopkg.in/check.v1

### [1.5.5](https://github.com/magiconair/properties/tags/v1.5.5) - 31 Jul 2015

 * [PR #6](https://github.com/magiconair/properties/pull/6): Add [Delete](http://godoc.org/github.com/magiconair/properties#Properties.Delete) method to remove keys including comments. (@gerbenjacobs)

### [1.5.4](https://github.com/magiconair/properties/tags/v1.5.4) - 23 Jun 2015

 * [Issue #5](https://github.com/magiconair/properties/issues/5): Allow disabling of property expansion [DisableExpansion](http://godoc.org/github.com/magiconair/properties#Properties.DisableExpansion). When property expansion is disabled Properties become a simple key/value store and don't check for circular references.

### [1.5.3](https://github.com/magiconair/properties/tags/v1.5.3) - 02 Jun 2015

 * [Issue #4](https://github.com/magiconair/properties/issues/4): Maintain key order in [Filter()](http://godoc.org/github.com/magiconair/properties#Properties.Filter), [FilterPrefix()](http://godoc.org/github.com/magiconair/properties#Properties.FilterPrefix) and [FilterRegexp()](http://godoc.org/github.com/magiconair/properties#Properties.FilterRegexp)

### [1.5.2](https://github.com/magiconair/properties/tags/v1.5.2) - 10 Apr 2015

 * [Issue #3](https://github.com/magiconair/properties/issues/3): Don't print comments in [WriteComment()](http://godoc.org/github.com/magiconair/properties#Properties.WriteComment) if they are all empty
 * Add clickable links to README

### [1.5.1](https://github.com/magiconair/properties/tags/v1.5.1) - 08 Dec 2014

 * Added [GetParsedDuration()](http://godoc.org/github.com/magiconair/properties#Properties.GetParsedDuration) and [MustGetParsedDuration()](http://godoc.org/github.com/magiconair/properties#Properties.MustGetParsedDuration) for values specified compatible with
   [time.ParseDuration()](http://golang.org/pkg/time/#ParseDuration).

### [1.5.0](https://github.com/magiconair/properties/tags/v1.5.0) - 18 Nov 2014

 * Added support for single and multi-line comments (reading, writing and updating)
 * The order of keys is now preserved
 * Calling [Set()](http://godoc.org/github.com/magiconair/properties#Properties.Set) with an empty key now silently ignores the call and does not create a new entry
 * Added a [MustSet()](http://godoc.org/github.com/magiconair/properties#Properties.MustSet) method
 * Migrated test library from launchpad.net/gocheck to [gopkg.in/check.v1](http://gopkg.in/check.v1)

### [1.4.2](https://github.com/magiconair/properties/tags/v1.4.2) - 15 Nov 2014

 * [Issue #2](https://github.com/magiconair/properties/issues/2): Fixed goroutine leak in parser which created two lexers but cleaned up only one

### [1.4.1](https://github.com/magiconair/properties/tags/v1.4.1) - 13 Nov 2014

 * [Issue #1](https://github.com/magiconair/properties/issues/1): Fixed bug in Keys() method which returned an empty string

### [1.4.0](https://github.com/magiconair/properties/tags/v1.4.0) - 23 Sep 2014

 * Added [Keys()](http://godoc.org/github.com/magiconair/properties#Properties.Keys) to get the keys
 * Added [Filter()](http://godoc.org/github.com/magiconair/properties#Properties.Filter), [FilterRegexp()](http://godoc.org/github.com/magiconair/properties#Properties.FilterRegexp) and [FilterPrefix()](http://godoc.org/github.com/magiconair/properties#Properties.FilterPrefix) to get a subset of the properties

### [1.3.0](https://github.com/magiconair/properties/tags/v1.3.0) - 18 Mar 2014

* Added support for time.Duration
* Made MustXXX() failure beha[ior configurable (log.Fatal, panic](https://github.com/magiconair/properties/tags/vior configurable (log.Fatal, panic) - custom)
* Changed default of MustXXX() failure from panic to log.Fatal

### [1.2.0](https://github.com/magiconair/properties/tags/v1.2.0) - 05 Mar 2014

* Added MustGet... functions
* Added support for int and uint with range checks on 32 bit platforms

### [1.1.0](https://github.com/magiconair/properties/tags/v1.1.0) - 20 Jan 2014

* Renamed from goproperties to properties
* Added support for expansion of environment vars in
  filenames and value expressions
* Fixed bug where value expressions were not at the
  start of the string

### [1.0.0](https://github.com/magiconair/properties/tags/v1.0.0) - 7 Jan 2014

* Initial release
