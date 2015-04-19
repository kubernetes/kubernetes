1.8.4 / 2014-09-23
==================

  * fix content encoding to be case-insensitive

1.8.3 / 2014-09-19
==================

  * deps: qs@2.2.4
    - Fix issue with object keys starting with numbers truncated

1.8.2 / 2014-09-15
==================

  * deps: depd@0.4.5

1.8.1 / 2014-09-07
==================

  * deps: media-typer@0.3.0
  * deps: type-is@~1.5.1

1.8.0 / 2014-09-05
==================

  * make empty-body-handling consistent between chunked requests
    - empty `json` produces `{}`
    - empty `raw` produces `new Buffer(0)`
    - empty `text` produces `''`
    - empty `urlencoded` produces `{}`
  * deps: qs@2.2.3
    - Fix issue where first empty value in array is discarded
  * deps: type-is@~1.5.0
    - fix `hasbody` to be true for `content-length: 0`

1.7.0 / 2014-09-01
==================

  * add `parameterLimit` option to `urlencoded` parser
  * change `urlencoded` extended array limit to 100
  * respond with 415 when over `parameterLimit` in `urlencoded`

1.6.7 / 2014-08-29
==================

  * deps: qs@2.2.2
    - Remove unnecessary cloning

1.6.6 / 2014-08-27
==================

  * deps: qs@2.2.0
    - Array parsing fix
    - Performance improvements

1.6.5 / 2014-08-16
==================

  * deps: on-finished@2.1.0

1.6.4 / 2014-08-14
==================

  * deps: qs@1.2.2

1.6.3 / 2014-08-10
==================

  * deps: qs@1.2.1

1.6.2 / 2014-08-07
==================

  * deps: qs@1.2.0
    - Fix parsing array of objects

1.6.1 / 2014-08-06
==================

  * deps: qs@1.1.0
    - Accept urlencoded square brackets
    - Accept empty values in implicit array notation

1.6.0 / 2014-08-05
==================

  * deps: qs@1.0.2
    - Complete rewrite
    - Limits array length to 20
    - Limits object depth to 5
    - Limits parameters to 1,000

1.5.2 / 2014-07-27
==================

  * deps: depd@0.4.4
    - Work-around v8 generating empty stack traces

1.5.1 / 2014-07-26
==================

  * deps: depd@0.4.3
    - Fix exception when global `Error.stackTraceLimit` is too low

1.5.0 / 2014-07-20
==================

  * deps: depd@0.4.2
    - Add `TRACE_DEPRECATION` environment variable
    - Remove non-standard grey color from color output
    - Support `--no-deprecation` argument
    - Support `--trace-deprecation` argument
  * deps: iconv-lite@0.4.4
    - Added encoding UTF-7
  * deps: raw-body@1.3.0
    - deps: iconv-lite@0.4.4
    - Added encoding UTF-7
    - Fix `Cannot switch to old mode now` error on Node.js 0.10+
  * deps: type-is@~1.3.2

1.4.3 / 2014-06-19
==================

  * deps: type-is@1.3.1
    - fix global variable leak

1.4.2 / 2014-06-19
==================

  * deps: type-is@1.3.0
    - improve type parsing

1.4.1 / 2014-06-19
==================

  * fix urlencoded extended deprecation message

1.4.0 / 2014-06-19
==================

  * add `text` parser
  * add `raw` parser
  * check accepted charset in content-type (accepts utf-8)
  * check accepted encoding in content-encoding (accepts identity)
  * deprecate `bodyParser()` middleware; use `.json()` and `.urlencoded()` as needed
  * deprecate `urlencoded()` without provided `extended` option
  * lazy-load urlencoded parsers
  * parsers split into files for reduced mem usage
  * support gzip and deflate bodies
    - set `inflate: false` to turn off
  * deps: raw-body@1.2.2
    - Support all encodings from `iconv-lite`

1.3.1 / 2014-06-11
==================

  * deps: type-is@1.2.1
    - Switch dependency from mime to mime-types@1.0.0

1.3.0 / 2014-05-31
==================

  * add `extended` option to urlencoded parser

1.2.2 / 2014-05-27
==================

  * deps: raw-body@1.1.6
    - assert stream encoding on node.js 0.8
    - assert stream encoding on node.js < 0.10.6
    - deps: bytes@1

1.2.1 / 2014-05-26
==================

  * invoke `next(err)` after request fully read
    - prevents hung responses and socket hang ups

1.2.0 / 2014-05-11
==================

  * add `verify` option
  * deps: type-is@1.2.0
    - support suffix matching

1.1.2 / 2014-05-11
==================

  * improve json parser speed

1.1.1 / 2014-05-11
==================

  * fix repeated limit parsing with every request

1.1.0 / 2014-05-10
==================

  * add `type` option
  * deps: pin for safety and consistency

1.0.2 / 2014-04-14
==================

  * use `type-is` module

1.0.1 / 2014-03-20
==================

  * lower default limits to 100kb
