
0.5.6 / 2013-04-09 
==================

  * fix empty key productions in parser

0.5.5 / 2013-03-20 
==================

  * output key= for null or undefined values. Closes #52

0.5.4 / 2013-03-15 
==================

  * fix ignoring of null / undefined. Closes #44

0.5.3 2012-12-09 
==================

  * add info to component.json
  * remove regular client-side ./querystring.js, fix component.json support

0.5.2 / 2012-11-14 
==================

  * fix uri encoding of non-plain object string values

0.5.1 / 2012-09-18 
==================

  * fix encoded `=`. Closes #43

0.5.0 / 2012-05-04 
==================

  * Added component support

0.4.2 / 2012-02-08 
==================

  * Fixed: ensure objects are created when appropriate not arrays [aheckmann]

0.4.1 / 2012-01-26 
==================

  * Fixed stringify()ing numbers. Closes #23

0.4.0 / 2011-11-21 
==================

  * Allow parsing of an existing object (for `bodyParser()`) [jackyz]
  * Replaced expresso with mocha

0.3.2 / 2011-11-08 
==================

  * Fixed global variable leak

0.3.1 / 2011-08-17 
==================

  * Added `try/catch` around malformed uri components
  * Add test coverage for Array native method bleed-though

0.3.0 / 2011-07-19 
==================

  * Allow `array[index]` and `object[property]` syntaxes [Aria Stewart]

0.2.0 / 2011-06-29 
==================

  * Added `qs.stringify()` [Cory Forsyth]

0.1.0 / 2011-04-13 
==================

  * Added jQuery-ish array support

0.0.7 / 2011-03-13 
==================

  * Fixed; handle empty string and `== null` in `qs.parse()` [dmit]
    allows for convenient `qs.parse(url.parse(str).query)`

0.0.6 / 2011-02-14 
==================

  * Fixed; support for implicit arrays

0.0.4 / 2011-02-09 
==================

  * Fixed `+` as a space

0.0.3 / 2011-02-08 
==================

  * Fixed case when right-hand value contains "]"

0.0.2 / 2011-02-07 
==================

  * Fixed "=" presence in key

0.0.1 / 2011-02-07 
==================

  * Initial release
