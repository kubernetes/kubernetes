# Lower Case First

[![NPM version][npm-image]][npm-url]
[![NPM downloads][downloads-image]][downloads-url]
[![Build status][travis-image]][travis-url]
[![Test coverage][coveralls-image]][coveralls-url]

Lower case the first character of a string.

Supports Unicode (non-ASCII characters) and non-string entities, such as objects with a `toString` property, numbers and booleans. Empty values (`null` and `undefined`) will result in an empty string.

## Installation

```
npm install lower-case-first --save
```

## Usage

```js
var lowerCaseFirst = require('lower-case-first')

lowerCaseFirst(null)     //=> ""
lowerCaseFirst('STRING') //=> "sTRING"
```

## License

MIT

[npm-image]: https://img.shields.io/npm/v/lower-case-first.svg?style=flat
[npm-url]: https://npmjs.org/package/lower-case-first
[downloads-image]: https://img.shields.io/npm/dm/lower-case-first.svg?style=flat
[downloads-url]: https://npmjs.org/package/lower-case-first
[travis-image]: https://img.shields.io/travis/blakeembrey/lower-case-first.svg?style=flat
[travis-url]: https://travis-ci.org/blakeembrey/lower-case-first
[coveralls-image]: https://img.shields.io/coveralls/blakeembrey/lower-case-first.svg?style=flat
[coveralls-url]: https://coveralls.io/r/blakeembrey/lower-case-first?branch=master
