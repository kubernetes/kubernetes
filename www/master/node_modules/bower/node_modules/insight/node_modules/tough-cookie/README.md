[RFC6265](http://tools.ietf.org/html/rfc6265) Cookies and CookieJar for Node.js

![Tough Cookie](http://www.goinstant.com.s3.amazonaws.com/tough-cookie.jpg)

[![Build Status](https://travis-ci.org/goinstant/node-cookie.png?branch=master)](https://travis-ci.org/goinstant/node-cookie)

[![NPM Stats](https://nodei.co/npm/tough-cookie.png?downloads=true&stars=true)](https://npmjs.org/package/tough-cookie)
![NPM Downloads](https://nodei.co/npm-dl/tough-cookie.png?months=9)

# Synopsis

``` javascript
var tough = require('tough-cookie'); // note: not 'cookie', 'cookies' or 'node-cookie'
var Cookie = tough.Cookie;
var cookie = Cookie.parse(header);
cookie.value = 'somethingdifferent';
header = cookie.toString();

var cookiejar = new tough.CookieJar();
cookiejar.setCookie(cookie, 'http://currentdomain.example.com/path', cb);
// ...
cookiejar.getCookies('http://example.com/otherpath',function(err,cookies) {
  res.headers['cookie'] = cookies.join('; ');
});
```

# Installation

It's _so_ easy!

`npm install tough-cookie`

Requires `punycode`, which should get installed automatically for you.  Note that node.js v0.6.2+ bundles punycode by default.

Why the name?  NPM modules `cookie`, `cookies` and `cookiejar` were already taken.

# API

tough
=====

Functions on the module you get from `require('tough-cookie')`.  All can be used as pure functions and don't need to be "bound".

parseDate(string[,strict])
-----------------

Parse a cookie date string into a `Date`.  Parses according to RFC6265 Section 5.1.1, not `Date.parse()`.  If strict is set to true then leading/trailing non-seperator characters around the time part will cause the parsing to fail (e.g. "Thu, 01 Jan 1970 00:00:010 GMT" has an extra trailing zero but Chrome, an assumedly RFC-compliant browser, treats this as valid).

formatDate(date)
----------------

Format a Date into a RFC1123 string (the RFC6265-recommended format).

canonicalDomain(str)
--------------------

Transforms a domain-name into a canonical domain-name.  The canonical domain-name is a trimmed, lowercased, stripped-of-leading-dot and optionally punycode-encoded domain-name (Section 5.1.2 of RFC6265).  For the most part, this function is idempotent (can be run again on its output without ill effects).

domainMatch(str,domStr[,canonicalize=true])
-------------------------------------------

Answers "does this real domain match the domain in a cookie?".  The `str` is the "current" domain-name and the `domStr` is the "cookie" domain-name.  Matches according to RFC6265 Section 5.1.3, but it helps to think of it as a "suffix match".

The `canonicalize` parameter will run the other two paramters through `canonicalDomain` or not.

defaultPath(path)
-----------------

Given a current request/response path, gives the Path apropriate for storing in a cookie.  This is basically the "directory" of a "file" in the path, but is specified by Section 5.1.4 of the RFC.

The `path` parameter MUST be _only_ the pathname part of a URI (i.e. excludes the hostname, query, fragment, etc.).  This is the `.pathname` property of node's `uri.parse()` output.

pathMatch(reqPath,cookiePath)
-----------------------------

Answers "does the request-path path-match a given cookie-path?" as per RFC6265 Section 5.1.4.  Returns a boolean.

This is essentially a prefix-match where `cookiePath` is a prefix of `reqPath`.

parse(header[,strict=false])
----------------------------

alias for `Cookie.parse(header[,strict])`

fromJSON(string)
----------------

alias for `Cookie.fromJSON(string)`

getPublicSuffix(hostname)
-------------------------

Returns the public suffix of this hostname.  The public suffix is the shortest domain-name upon which a cookie can be set.  Returns `null` if the hostname cannot have cookies set for it.

For example: `www.example.com` and `www.subdomain.example.com` both have public suffix `example.com`.

For further information, see http://publicsuffix.org/.  This module derives its list from that site.

cookieCompare(a,b)
------------------

For use with `.sort()`, sorts a list of cookies into the recommended order given in the RFC (Section 5.4 step 2).  Longest `.path`s go first, then sorted oldest to youngest.

``` javascript
var cookies = [ /* unsorted array of Cookie objects */ ];
cookies = cookies.sort(cookieCompare);
```

permuteDomain(domain)
---------------------

Generates a list of all possible domains that `domainMatch()` the parameter.  May be handy for implementing cookie stores.


permutePath(path)
-----------------

Generates a list of all possible paths that `pathMatch()` the parameter.  May be handy for implementing cookie stores.

Cookie
======

Cookie.parse(header[,strict=false])
-----------------------------------

Parses a single Cookie or Set-Cookie HTTP header into a `Cookie` object.  Returns `undefined` if the string can't be parsed.  If in strict mode, returns `undefined` if the cookie doesn't follow the guidelines in section 4 of RFC6265.  Generally speaking, strict mode can be used to validate your own generated Set-Cookie headers, but acting as a client you want to be lenient and leave strict mode off.

Here's how to process the Set-Cookie header(s) on a node HTTP/HTTPS response:

``` javascript
if (res.headers['set-cookie'] instanceof Array)
  cookies = res.headers['set-cookie'].map(function (c) { return (Cookie.parse(c)); });
else
  cookies = [Cookie.parse(res.headers['set-cookie'])];
```

Cookie.fromJSON(string)
-----------------------

Convert a JSON string to a `Cookie` object.  Does a `JSON.parse()` and converts the `.created`, `.lastAccessed` and `.expires` properties into `Date` objects.

Properties
==========

  * _key_ - string - the name or key of the cookie (default "")
  * _value_ - string - the value of the cookie (default "")
  * _expires_ - `Date` - if set, the `Expires=` attribute of the cookie (defaults to the string `"Infinity"`). See `setExpires()`
  * _maxAge_ - seconds - if set, the `Max-Age=` attribute _in seconds_ of the cookie.  May also be set to strings `"Infinity"` and `"-Infinity"` for non-expiry and immediate-expiry, respectively.  See `setMaxAge()`
  * _domain_ - string - the `Domain=` attribute of the cookie
  * _path_ - string - the `Path=` of the cookie
  * _secure_ - boolean - the `Secure` cookie flag
  * _httpOnly_ - boolean - the `HttpOnly` cookie flag
  * _extensions_ - `Array` - any unrecognized cookie attributes as strings (even if equal-signs inside)

After a cookie has been passed through `CookieJar.setCookie()` it will have the following additional attributes:

  * _hostOnly_ - boolean - is this a host-only cookie (i.e. no Domain field was set, but was instead implied)
  * _pathIsDefault_ - boolean - if true, there was no Path field on the cookie and `defaultPath()` was used to derive one.
  * _created_ - `Date` - when this cookie was added to the jar
  * _lastAccessed_ - `Date` - last time the cookie got accessed. Will affect cookie cleaning once implemented.  Using `cookiejar.getCookies(...)` will update this attribute.

Construction([{options}])
------------

Receives an options object that can contain any Cookie properties, uses the default for unspecified properties.

.toString()
-----------

encode to a Set-Cookie header value.  The Expires cookie field is set using `formatDate()`, but is omitted entirely if `.expires` is `Infinity`.

.cookieString()
---------------

encode to a Cookie header value (i.e. the `.key` and `.value` properties joined with '=').

.setExpires(String)
-------------------

sets the expiry based on a date-string passed through `parseDate()`.  If parseDate returns `null` (i.e. can't parse this date string), `.expires` is set to `"Infinity"` (a string) is set.

.setMaxAge(number)
-------------------

sets the maxAge in seconds.  Coerces `-Infinity` to `"-Infinity"` and `Infinity` to `"Infinity"` so it JSON serializes correctly.

.expiryTime([now=Date.now()])
-----------------------------

.expiryDate([now=Date.now()])
-----------------------------

expiryTime() Computes the absolute unix-epoch milliseconds that this cookie expires. expiryDate() works similarly, except it returns a `Date` object.  Note that in both cases the `now` parameter should be milliseconds.

Max-Age takes precedence over Expires (as per the RFC). The `.created` attribute -- or, by default, the `now` paramter -- is used to offset the `.maxAge` attribute.

If Expires (`.expires`) is set, that's returned.

Otherwise, `expiryTime()` returns `Infinity` and `expiryDate()` returns a `Date` object for "Tue, 19 Jan 2038 03:14:07 GMT" (latest date that can be expressed by a 32-bit `time_t`; the common limit for most user-agents).

.TTL([now=Date.now()])
---------

compute the TTL relative to `now` (milliseconds).  The same precedence rules as for `expiryTime`/`expiryDate` apply.

The "number" `Infinity` is returned for cookies without an explicit expiry and `0` is returned if the cookie is expired.  Otherwise a time-to-live in milliseconds is returned.

.canonicalizedDoman()
---------------------

.cdomain()
----------

return the canonicalized `.domain` field.  This is lower-cased and punycode (RFC3490) encoded if the domain has any non-ASCII characters.

.validate()
-----------

Status: *IN PROGRESS*. Works for a few things, but is by no means comprehensive.

validates cookie attributes for semantic correctness.  Useful for "lint" checking any Set-Cookie headers you generate.  For now, it returns a boolean, but eventually could return a reason string -- you can future-proof with this construct:

``` javascript
if (cookie.validate() === true) {
  // it's tasty
} else {
  // yuck!
}
```

CookieJar
=========

Construction([store = new MemoryCookieStore()][, rejectPublicSuffixes])
------------

Simply use `new CookieJar()`.  If you'd like to use a custom store, pass that to the constructor otherwise a `MemoryCookieStore` will be created and used.


Attributes
----------

  * _rejectPublicSuffixes_ - boolean - reject cookies with domains like "com" and "co.uk" (default: `true`)

Since eventually this module would like to support database/remote/etc. CookieJars, continuation passing style is used for CookieJar methods.

.setCookie(cookieOrString, currentUrl, [{options},] cb(err,cookie))
-------------------------------------------------------------------

Attempt to set the cookie in the cookie jar.  If the operation fails, an error will be given to the callback `cb`, otherwise the cookie is passed through.  The cookie will have updated `.created`, `.lastAccessed` and `.hostOnly` properties.

The `options` object can be omitted and can have the following properties:

  * _http_ - boolean - default `true` - indicates if this is an HTTP or non-HTTP API.  Affects HttpOnly cookies.
  * _secure_ - boolean - autodetect from url - indicates if this is a "Secure" API.  If the currentUrl starts with `https:` or `wss:` then this is defaulted to `true`, otherwise `false`.
  * _now_ - Date - default `new Date()` - what to use for the creation/access time of cookies
  * _strict_ - boolean - default `false` - perform extra checks
  * _ignoreError_ - boolean - default `false` - silently ignore things like parse errors and invalid domains.  CookieStore errors aren't ignored by this option.

As per the RFC, the `.hostOnly` property is set if there was no "Domain=" parameter in the cookie string (or `.domain` was null on the Cookie object).  The `.domain` property is set to the fully-qualified hostname of `currentUrl` in this case.  Matching this cookie requires an exact hostname match (not a `domainMatch` as per usual).

.setCookieSync(cookieOrString, currentUrl, [{options}])
-------------------------------------------------------

Synchronous version of `setCookie`; only works with synchronous stores (e.g. the default `MemoryCookieStore`).

.storeCookie(cookie, [{options},] cb(err,cookie))
-------------------------------------------------

__REMOVED__ removed in lieu of the CookieStore API below

.getCookies(currentUrl, [{options},] cb(err,cookies))
-----------------------------------------------------

Retrieve the list of cookies that can be sent in a Cookie header for the current url.

If an error is encountered, that's passed as `err` to the callback, otherwise an `Array` of `Cookie` objects is passed.  The array is sorted with `cookieCompare()` unless the `{sort:false}` option is given.

The `options` object can be omitted and can have the following properties:

  * _http_ - boolean - default `true` - indicates if this is an HTTP or non-HTTP API.  Affects HttpOnly cookies.
  * _secure_ - boolean - autodetect from url - indicates if this is a "Secure" API.  If the currentUrl starts with `https:` or `wss:` then this is defaulted to `true`, otherwise `false`.
  * _now_ - Date - default `new Date()` - what to use for the creation/access time of cookies
  * _expire_ - boolean - default `true` - perform expiry-time checking of cookies and asynchronously remove expired cookies from the store.  Using `false` will return expired cookies and **not** remove them from the store (which is useful for replaying Set-Cookie headers, potentially).
  * _allPaths_ - boolean - default `false` - if `true`, do not scope cookies by path. The default uses RFC-compliant path scoping. **Note**: may not be supported by the CookieStore `fetchCookies` function (the default MemoryCookieStore supports it).

The `.lastAccessed` property of the returned cookies will have been updated.

.getCookiesSync(currentUrl, [{options}])
----------------------------------------

Synchronous version of `getCookies`; only works with synchronous stores (e.g. the default `MemoryCookieStore`).

.getCookieString(...)
---------------------

Accepts the same options as `.getCookies()` but passes a string suitable for a Cookie header rather than an array to the callback.  Simply maps the `Cookie` array via `.cookieString()`.

.getCookieStringSync(...)
-------------------------

Synchronous version of `getCookieString`; only works with synchronous stores (e.g. the default `MemoryCookieStore`).

.getSetCookieStrings(...)
-------------------------

Returns an array of strings suitable for **Set-Cookie** headers. Accepts the same options as `.getCookies()`.  Simply maps the cookie array via `.toString()`.

.getSetCookieStringsSync(...)
-----------------------------

Synchronous version of `getSetCookieStrings`; only works with synchronous stores (e.g. the default `MemoryCookieStore`).

Store
=====

Base class for CookieJar stores.

# CookieStore API

The storage model for each `CookieJar` instance can be replaced with a custom implementation.  The default is `MemoryCookieStore` which can be found in the `lib/memstore.js` file.  The API uses continuation-passing-style to allow for asynchronous stores.

Stores should inherit from the base `Store` class, which is available as `require('tough-cookie').Store`.  Stores are asynchronous by default, but if `store.synchronous` is set, then the `*Sync` methods on the CookieJar can be used.

All `domain` parameters will have been normalized before calling.

The Cookie store must have all of the following methods.

store.findCookie(domain, path, key, cb(err,cookie))
---------------------------------------------------

Retrieve a cookie with the given domain, path and key (a.k.a. name).  The RFC maintains that exactly one of these cookies should exist in a store.  If the store is using versioning, this means that the latest/newest such cookie should be returned.

Callback takes an error and the resulting `Cookie` object.  If no cookie is found then `null` MUST be passed instead (i.e. not an error).

store.findCookies(domain, path, cb(err,cookies))
------------------------------------------------

Locates cookies matching the given domain and path.  This is most often called in the context of `cookiejar.getCookies()` above.

If no cookies are found, the callback MUST be passed an empty array.

The resulting list will be checked for applicability to the current request according to the RFC (domain-match, path-match, http-only-flag, secure-flag, expiry, etc.), so it's OK to use an optimistic search algorithm when implementing this method.  However, the search algorithm used SHOULD try to find cookies that `domainMatch()` the domain and `pathMatch()` the path in order to limit the amount of checking that needs to be done.

As of version 0.9.12, the `allPaths` option to `cookiejar.getCookies()` above will cause the path here to be `null`.  If the path is `null`, path-matching MUST NOT be performed (i.e. domain-matching only).

store.putCookie(cookie, cb(err))
--------------------------------

Adds a new cookie to the store.  The implementation SHOULD replace any existing cookie with the same `.domain`, `.path`, and `.key` properties -- depending on the nature of the implementation, it's possible that between the call to `fetchCookie` and `putCookie` that a duplicate `putCookie` can occur.

The `cookie` object MUST NOT be modified; the caller will have already updated the `.creation` and `.lastAccessed` properties.

Pass an error if the cookie cannot be stored.

store.updateCookie(oldCookie, newCookie, cb(err))
-------------------------------------------------

Update an existing cookie.  The implementation MUST update the `.value` for a cookie with the same `domain`, `.path` and `.key`.  The implementation SHOULD check that the old value in the store is equivalent to `oldCookie` - how the conflict is resolved is up to the store.

The `.lastAccessed` property will always be different between the two objects and `.created` will always be the same.  Stores MAY ignore or defer the `.lastAccessed` change at the cost of affecting how cookies are sorted (or selected for deletion).

Stores may wish to optimize changing the `.value` of the cookie in the store versus storing a new cookie.  If the implementation doesn't define this method a stub that calls `putCookie(newCookie,cb)` will be added to the store object.

The `newCookie` and `oldCookie` objects MUST NOT be modified.

Pass an error if the newCookie cannot be stored.

store.removeCookie(domain, path, key, cb(err))
----------------------------------------------

Remove a cookie from the store (see notes on `findCookie` about the uniqueness constraint).

The implementation MUST NOT pass an error if the cookie doesn't exist; only pass an error due to the failure to remove an existing cookie.

store.removeCookies(domain, path, cb(err))
------------------------------------------

Removes matching cookies from the store.  The `path` paramter is optional, and if missing means all paths in a domain should be removed.

Pass an error ONLY if removing any existing cookies failed.

# TODO

  * _full_ RFC5890/RFC5891 canonicalization for domains in `cdomain()`
    * the optional `punycode` requirement implements RFC3492, but RFC6265 requires RFC5891
  * better tests for `validate()`?

# Copyright and License

(tl;dr: MIT with some MPL/1.1)

Copyright 2012- GoInstant, Inc. and other contributors. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

Portions may be licensed under different licenses (in particular public-suffix.txt is MPL/1.1); please read the LICENSE file for full details.
