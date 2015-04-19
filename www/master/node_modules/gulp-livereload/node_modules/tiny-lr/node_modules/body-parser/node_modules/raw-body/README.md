# raw-body

[![NPM version](https://badge.fury.io/js/raw-body.svg)](http://badge.fury.io/js/raw-body)
[![Build Status](https://travis-ci.org/stream-utils/raw-body.svg?branch=master)](https://travis-ci.org/stream-utils/raw-body)
[![Coverage Status](https://img.shields.io/coveralls/stream-utils/raw-body.svg?branch=master)](https://coveralls.io/r/stream-utils/raw-body)

Gets the entire buffer of a stream either as a `Buffer` or a string.
Validates the stream's length against an expected length and maximum limit.
Ideal for parsing request bodies.

## API

```js
var getRawBody = require('raw-body')
var typer      = require('media-typer')

app.use(function (req, res, next) {
  getRawBody(req, {
    length: req.headers['content-length'],
    limit: '1mb',
    encoding: typer.parse(req.headers['content-type']).parameters.charset
  }, function (err, string) {
    if (err)
      return next(err)

    req.text = string
    next()
  })
})
```

or in a Koa generator:

```js
app.use(function* (next) {
  var string = yield getRawBody(this.req, {
    length: this.length,
    limit: '1mb',
    encoding: this.charset
  })
})
```

### getRawBody(stream, [options], [callback])

Returns a thunk for yielding with generators.

Options:

- `length` - The length length of the stream.
  If the contents of the stream do not add up to this length,
  an `400` error code is returned.
- `limit` - The byte limit of the body.
  If the body ends up being larger than this limit,
  a `413` error code is returned.
- `encoding` - The requested encoding.
  By default, a `Buffer` instance will be returned.
  Most likely, you want `utf8`.
  You can use any type of encoding supported by [iconv-lite](https://www.npmjs.org/package/iconv-lite#readme).

You can also pass a string in place of options to just specify the encoding.

`callback(err, res)`:

- `err` - the following attributes will be defined if applicable:

    - `limit` - the limit in bytes
    - `length` and `expected` - the expected length of the stream
    - `received` - the received bytes
    - `encoding` - the invalid encoding
    - `status` and `statusCode` - the corresponding status code for the error
    - `type` - either `entity.too.large`, `request.size.invalid`, `stream.encoding.set`, or `encoding.unsupported`

- `res` - the result, either as a `String` if an encoding was set or a `Buffer` otherwise.

If an error occurs, the stream will be paused, everything unpiped,
and you are responsible for correctly disposing the stream.
For HTTP requests, no handling is required if you send a response.
For streams that use file descriptors, you should `stream.destroy()` or `stream.close()` to prevent leaks.

## License

The MIT License (MIT)

Copyright (c) 2013 Jonathan Ong me@jongleberry.com

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
