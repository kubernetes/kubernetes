1.5.7 / 2015-02-09
==================

  * fix argument reassignment
  * deps: mime-types@~2.0.9
    - Add new mime types

1.5.6 / 2015-01-29
==================

  * deps: mime-types@~2.0.8
    - Add new mime types

1.5.5 / 2014-12-30
==================

  * deps: mime-types@~2.0.7
    - Add new mime types
    - Fix missing extensions
    - Fix various invalid MIME type entries
    - Remove example template MIME types
    - deps: mime-db@~1.5.0

1.5.4 / 2014-12-10
==================

  * deps: mime-types@~2.0.4
    - Add new mime types
    - deps: mime-db@~1.3.0

1.5.3 / 2014-11-09
==================

  * deps: mime-types@~2.0.3
    - Add new mime types
    - deps: mime-db@~1.2.0

1.5.2 / 2014-09-28
==================

  * deps: mime-types@~2.0.2
    - Add new mime types
    - deps: mime-db@~1.1.0

1.5.1 / 2014-09-07
==================

  * Support Node.js 0.6
  * deps: media-typer@0.3.0
  * deps: mime-types@~2.0.1
    - Support Node.js 0.6

1.5.0 / 2014-09-05
==================

 * fix `hasbody` to be true for `content-length: 0`

1.4.0 / 2014-09-02
==================

 * update mime-types

1.3.2 / 2014-06-24
==================

 * use `~` range on mime-types

1.3.1 / 2014-06-19
==================

 * fix global variable leak

1.3.0 / 2014-06-19
==================

 * improve type parsing

   - invalid media type never matches
   - media type not case-sensitive
   - extra LWS does not affect results

1.2.2 / 2014-06-19
==================

 * fix behavior on unknown type argument

1.2.1 / 2014-06-03
==================

 * switch dependency from `mime` to `mime-types@1.0.0`

1.2.0 / 2014-05-11
==================

 * support suffix matching:

   - `+json` matches `application/vnd+json`
   - `*/vnd+json` matches `application/vnd+json`
   - `application/*+json` matches `application/vnd+json`

1.1.0 / 2014-04-12
==================

 * add non-array values support
 * expose internal utilities:

   - `.is()`
   - `.hasBody()`
   - `.normalize()`
   - `.match()`

1.0.1 / 2014-03-30
==================

 * add `multipart` as a shorthand
