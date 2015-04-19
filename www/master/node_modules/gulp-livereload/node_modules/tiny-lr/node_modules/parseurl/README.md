# parseurl

[![NPM version](https://badge.fury.io/js/parseurl.svg)](http://badge.fury.io/js/parseurl)
[![Build Status](https://travis-ci.org/expressjs/parseurl.svg?branch=master)](https://travis-ci.org/expressjs/parseurl)
[![Coverage Status](https://img.shields.io/coveralls/expressjs/parseurl.svg?branch=master)](https://coveralls.io/r/expressjs/parseurl)

Parse a URL with memoization.

## Install

```bash
$ npm install parseurl
```

## API

```js
var parseurl = require('parseurl')
```

### parseurl(req)

Parse the URL of the given request object (looks at the `req.url` property)
and return the result. The result is the same as `url.parse` in Node.js core.
Calling this function multiple times on the same `req` where `req.url` does
not change will return a cached parsed object, rather than parsing again.

### parseurl.original(req)

Parse the original URL of the given request object and return the result.
This works by trying to parse `req.originalUrl` if it is a string, otherwise
parses `req.url`. The result is the same as `url.parse` in Node.js core.
Calling this function multiple times on the same `req` where `req.originalUrl`
does not change will return a cached parsed object, rather than parsing again.

## Benchmark

```bash
$ npm run-script bench

> parseurl@1.3.0 bench nodejs-parseurl
> node benchmark/index.js

> node benchmark/fullurl.js

  Parsing URL "http://localhost:8888/foo/bar?user=tj&pet=fluffy"

  1 test completed.
  2 tests completed.
  3 tests completed.

  fasturl   x 1,290,780 ops/sec ±0.46% (195 runs sampled)
  nativeurl x    56,401 ops/sec ±0.22% (196 runs sampled)
  parseurl  x    55,231 ops/sec ±0.22% (194 runs sampled)

> node benchmark/pathquery.js

  Parsing URL "/foo/bar?user=tj&pet=fluffy"

  1 test completed.
  2 tests completed.
  3 tests completed.

  fasturl   x 1,986,668 ops/sec ±0.27% (190 runs sampled)
  nativeurl x    98,740 ops/sec ±0.21% (195 runs sampled)
  parseurl  x 2,628,171 ops/sec ±0.36% (195 runs sampled)

> node benchmark/samerequest.js

  Parsing URL "/foo/bar?user=tj&pet=fluffy" on same request object

  1 test completed.
  2 tests completed.
  3 tests completed.

  fasturl   x  2,184,468 ops/sec ±0.40% (194 runs sampled)
  nativeurl x     99,437 ops/sec ±0.71% (194 runs sampled)
  parseurl  x 10,498,005 ops/sec ±0.61% (186 runs sampled)

> node benchmark/simplepath.js

  Parsing URL "/foo/bar"

  1 test completed.
  2 tests completed.
  3 tests completed.

  fasturl   x 4,535,825 ops/sec ±0.27% (191 runs sampled)
  nativeurl x    98,769 ops/sec ±0.54% (191 runs sampled)
  parseurl  x 4,164,865 ops/sec ±0.34% (192 runs sampled)

> node benchmark/slash.js

  Parsing URL "/"

  1 test completed.
  2 tests completed.
  3 tests completed.

  fasturl   x 4,908,405 ops/sec ±0.42% (191 runs sampled)
  nativeurl x   100,945 ops/sec ±0.59% (188 runs sampled)
  parseurl  x 4,333,208 ops/sec ±0.27% (194 runs sampled)
```

## License

  [MIT](LICENSE)
