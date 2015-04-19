# stream-consume

A node module ensures a Readable stream continues flowing if it's not piped to
another destination.

	npm install stream-consume

## Usage

Simply pass a stream to `stream-consume`.
Both legacy streams and streams2 are supported.

``` js
var consume = require('stream-consume');

consume(readableStream);
```

## Details

Only Readable streams are processed (as determined by presence of `readable`
property and a `resume` property that is a function). If called with anything
else, it's a NOP.

For a streams2 stream (as determined by presence of a `_readableState`
property), nothing is done if the stream has already been piped to at least
one other destination.

`resume()` is used to cause the stream to continue flowing.

## License

The MIT License (MIT)

Copyright (c) 2014 Aron Nopanen

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
