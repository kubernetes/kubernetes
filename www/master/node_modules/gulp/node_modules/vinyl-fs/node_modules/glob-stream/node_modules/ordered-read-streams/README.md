# ordered-read-streams [![NPM version](https://badge.fury.io/js/ordered-read-streams.png)](http://badge.fury.io/js/ordered-read-streams) [![Build Status](https://travis-ci.org/armed/ordered-read-streams.png?branch=master)](https://travis-ci.org/armed/ordered-read-streams)

Combines array of streams into one read stream in strict order.

## Installation

`npm install ordered-read-streams`

## Overview

`ordered-read-streams` handles all data/errors from input streams in parallel, but emits data/errors in strict order in which streams are passed to constructor. This is `objectMode = true` stream.

## Example

```js
var through = require('through2');
var Ordered = require('ordered-read-streams');

var s1 = through.obj(function (data, enc, next) {
  var self = this;
  setTimeout(function () {
    self.push(data);
    next();
  }, 200)
});
var s2 = through.obj(function (data, enc, next) {
  var self = this;
  setTimeout(function () {
    self.push(data);
    next();
  }, 30)
});
var s3 = through.obj(function (data, enc, next) {
  var self = this;
  setTimeout(function () {
    self.push(data);
    next();
  }, 100)
});

var streams = new Ordered([s1, s2, s3]);
streams.on('data', function (data) {
  console.log(data);
})

s1.write('stream 1');
s1.end();

s2.write('stream 2');
s2.end();

s3.write('stream 3');
s3.end();
```
Ouput will be:

```
stream 1
stream 2
stream 3
```

## Licence

MIT
