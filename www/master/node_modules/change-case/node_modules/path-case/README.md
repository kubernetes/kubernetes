# Path Case

[![NPM version][npm-image]][npm-url]
[![NPM downloads][downloads-image]][downloads-url]
[![Build status][travis-image]][travis-url]
[![Test coverage][coveralls-image]][coveralls-url]

Path case a string.

Supports Unicode (non-ASCII characters) and non-string entities, such as objects with a `toString` property, numbers and booleans. Empty values (`null` and `undefined`) will result in an empty string.

## Installation

```bash
npm install path-case --save
```

## Usage

```javascript
var pathCase = require('path-case');

pathCase('string');        //=> "string"
pathCase('camelCase');     //=> "camel/case"
pathCase('sentence case'); //=> "sentence/case"

pathCase('MY STRING', 'tr'); //=> "my.strıng"
```

## License

MIT

[npm-image]: https://img.shields.io/npm/v/path-case.svg?style=flat
[npm-url]: https://npmjs.org/package/path-case
[downloads-image]: https://img.shields.io/npm/dm/path-case.svg?style=flat
[downloads-url]: https://npmjs.org/package/path-case
[travis-image]: https://img.shields.io/travis/blakeembrey/path-case.svg?style=flat
[travis-url]: https://travis-ci.org/blakeembrey/path-case
[coveralls-image]: https://img.shields.io/coveralls/blakeembrey/path-case.svg?style=flat
[coveralls-url]: https://coveralls.io/r/blakeembrey/path-case?branch=master
