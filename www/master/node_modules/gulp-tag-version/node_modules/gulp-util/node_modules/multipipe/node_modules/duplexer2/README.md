duplexer2 [![build status](https://travis-ci.org/deoxxa/duplexer2.png)](https://travis-ci.org/deoxxa/fork)
=========

Like duplexer (http://npm.im/duplexer) but using streams2.

Overview
--------

duplexer2 is a reimplementation of [duplexer](http://npm.im/duplexer) using the
readable-stream API which is standard in node as of v0.10. Everything largely
works the same.

Installation
------------

Available via [npm](http://npmjs.org/):

> $ npm install duplexer2

Or via git:

> $ git clone git://github.com/deoxxa/duplexer2.git node_modules/duplexer2

API
---

**duplexer2**

Creates a new `DuplexWrapper` object, which is the actual class that implements
most of the fun stuff. All that fun stuff is hidden. DON'T LOOK.

```javascript
duplexer2([options], writable, readable)
```

```javascript
var duplex = duplexer2(new stream.Writable(), new stream.Readable());
```

Arguments

* __options__ - an object specifying the regular `stream.Duplex` options, as
  well as the properties described below.
* __writable__ - a writable stream
* __readable__ - a readable stream

Options

* __bubbleErrors__ - a boolean value that specifies whether to bubble errors
  from the underlying readable/writable streams. Default is `true`.

Example
-------

Also see [example.js](https://github.com/deoxxa/duplexer2/blob/master/example.js).

Code:

```javascript
var stream = require("stream");

var duplexer2 = require("duplexer2");

var writable = new stream.Writable({objectMode: true}),
    readable = new stream.Readable({objectMode: true});

writable._write = function _write(input, encoding, done) {
  if (readable.push(input)) {
    return done();
  } else {
    readable.once("drain", done);
  }
};

readable._read = function _read(n) {
  // no-op
};

// simulate the readable thing closing after a bit
writable.once("finish", function() {
  setTimeout(function() {
    readable.push(null);
  }, 500);
});

var duplex = duplexer2(writable, readable);

duplex.on("data", function(e) {
  console.log("got data", JSON.stringify(e));
});

duplex.on("finish", function() {
  console.log("got finish event");
});

duplex.on("end", function() {
  console.log("got end event");
});

duplex.write("oh, hi there", function() {
  console.log("finished writing");
});

duplex.end(function() {
  console.log("finished ending");
});
```

Output:

```
got data "oh, hi there"
finished writing
got finish event
finished ending
got end event
```

License
-------

3-clause BSD. A copy is included with the source.

Contact
-------

* GitHub ([deoxxa](http://github.com/deoxxa))
* Twitter ([@deoxxa](http://twitter.com/deoxxa))
* Email ([deoxxa@fknsrs.biz](mailto:deoxxa@fknsrs.biz))
