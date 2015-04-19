# Upper Case First

[![NPM version][npm-image]][npm-url]
[![NPM downloads][downloads-image]][downloads-url]
[![Build status][travis-image]][travis-url]
[![Test coverage][coveralls-image]][coveralls-url]

Upper case the first character of a string.

Supports Unicode (non-ASCII characters) and non-string entities, such as objects with a `toString` property, numbers and booleans. Empty values (`null` and `undefined`) will result in an empty string.

## Installation

```
npm install upper-case-first --save
```

## Usage

```js
var upperCaseFirst = require('upper-case-first')

upperCaseFirst(null)     //=> ""
upperCaseFirst('string') //=> "String"
```

## License

MIT

[npm-image]: https://img.shields.io/npm/v/upper-case-first.svg?style=flat
[npm-url]: https://npmjs.org/package/upper-case-first
[downloads-image]: https://img.shields.io/npm/dm/upper-case-first.svg?style=flat
[downloads-url]: https://npmjs.org/package/upper-case-first
[travis-image]: https://img.shields.io/travis/blakeembrey/upper-case-first.svg?style=flat
[travis-url]: https://travis-ci.org/blakeembrey/upper-case-first
[coveralls-image]: https://img.shields.io/coveralls/blakeembrey/upper-case-first.svg?style=flat
[coveralls-url]: https://coveralls.io/r/blakeembrey/upper-case-first?branch=master
