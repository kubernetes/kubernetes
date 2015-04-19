# multipipe

A better `Stream#pipe` that creates duplex streams and lets you handle errors in one place.

[![build status](https://secure.travis-ci.org/segmentio/multipipe.png)](http://travis-ci.org/segmentio/multipipe)

## Example

```js
var pipe = require('multipipe');

// pipe streams
var stream = pipe(streamA, streamB, streamC);

// centralized error handling
stream.on('error', fn);

// creates a new stream
source.pipe(stream).pipe(dest);

// optional callback on finish or error
pipe(streamA, streamB, streamC, function(err){
  // ...
});
```

## Duplex streams

  Write to the pipe and you'll really write to the first stream, read from the pipe and you'll read from the last stream.

```js
var stream = pipe(a, b, c);

source
  .pipe(stream)
  .pipe(destination);
```

  In this example the flow of data is:

  * source ->
  * a ->
  * b ->
  * c ->
  * destination

## Error handling

  Each `pipe` forwards the errors the streams it wraps emit, so you have one central place to handle errors:

```js
var stream = pipe(a, b, c);

stream.on('error', function(err){
  // called three times
});

a.emit('error', new Error);
b.emit('error', new Error);
c.emit('error', new Error);
```

## API

### pipe(stream, ...)

Pass a variable number of streams and each will be piped to the next one.

A stream will be returned that wraps passed in streams in a way that errors will be forwarded and you can write to and/or read from it.

Pass a function as last argument to be called on `error` or `finish` of the last stream.

## Installation

```bash
$ npm install multipipe
```

## License

The MIT License (MIT)

Copyright (c) 2014 Segment.io Inc. &lt;friends@segment.io&gt;
Copyright (c) 2014 Julian Gruber &lt;julian@juliangruber.com&gt;

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
