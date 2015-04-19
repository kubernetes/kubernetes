1.3.0 / 2014-07-20
==================

  * Fully unpipe the stream on error
    - Fixes `Cannot switch to old mode now` error on Node.js 0.10+

1.2.3 / 2014-07-20
==================

  * deps: iconv-lite@0.4.4
    - Added encoding UTF-7

1.2.2 / 2014-06-19
==================

  * Send invalid encoding error to callback

1.2.1 / 2014-06-15
==================

  * deps: iconv-lite@0.4.3
    - Added encodings UTF-16BE and UTF-16 with BOM

1.2.0 / 2014-06-13
==================

  * Passing string as `options` interpreted as encoding
  * Support all encodings from `iconv-lite`

1.1.7 / 2014-06-12
==================

  * use `string_decoder` module from npm

1.1.6 / 2014-05-27
==================

  * check encoding for old streams1
  * support node.js < 0.10.6

1.1.5 / 2014-05-14
==================

  * bump bytes

1.1.4 / 2014-04-19
==================

  * allow true as an option
  * bump bytes

1.1.3 / 2014-03-02
==================

  * fix case when length=null

1.1.2 / 2013-12-01
==================

  * be less strict on state.encoding check

1.1.1 / 2013-11-27
==================

  * add engines

1.1.0 / 2013-11-27
==================

  * add err.statusCode and err.type
  * allow for encoding option to be true
  * pause the stream instead of dumping on error
  * throw if the stream's encoding is set

1.0.1 / 2013-11-19
==================

  * dont support streams1, throw if dev set encoding

1.0.0 / 2013-11-17
==================

  * rename `expected` option to `length`

0.2.0 / 2013-11-15
==================

  * republish

0.1.1 / 2013-11-15
==================

  * use bytes

0.1.0 / 2013-11-11
==================

  * generator support

0.0.3 / 2013-10-10
==================

  * update repo

0.0.2 / 2013-09-14
==================

  * dump stream on bad headers
  * listen to events after defining received and buffers

0.0.1 / 2013-09-14
==================

  * Initial release
