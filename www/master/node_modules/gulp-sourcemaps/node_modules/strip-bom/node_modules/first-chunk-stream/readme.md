# first-chunk-stream [![Build Status](https://travis-ci.org/sindresorhus/first-chunk-stream.svg?branch=master)](https://travis-ci.org/sindresorhus/first-chunk-stream)

> Transform the first chunk in a stream

Useful if you want to do something to the first chunk.

You can also set the minimum size of that chunk.


## Install

```sh
$ npm install --save first-chunk-stream
```


## Usage

```js
var fs = require('fs');
var concat = require('concat-stream');
var firstChunk = require('first-chunk-stream');

// unicorn.txt => unicorn rainbow
// `highWaterMark: 1` means it will only read 1 byte at the time
fs.createReadStream('unicorn.txt', {highWaterMark: 1})
	.pipe(firstChunk({minSize: 7}, function (chunk, enc, cb) {
		this.push(chunk.toUpperCase());
		cb();
	}))
	.pipe(concat(function (data) {
		console.log(data);
		//=> UNICORN rainbow
	}));
```


## API

### firstChunk([options], transform)

#### options.minSize

Type: `number`

The minimum size of the first chunk.

#### transform(chunk, encoding, callback)

*Required*  
Type: `function`

The [function](http://nodejs.org/docs/latest/api/stream.html#stream_transform_transform_chunk_encoding_callback) that gets the first chunk.

### firstChunk.ctor()

Instead of returning a [stream.Transform](http://nodejs.org/docs/latest/api/stream.html#stream_class_stream_transform_1) instance, `firstChunk.ctor()` returns a constructor for a custom Transform. This is useful when you want to use the same transform logic in multiple instances.


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
