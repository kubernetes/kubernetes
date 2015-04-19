# Dot Case

[![NPM version][npm-image]][npm-url]
[![NPM downloads][downloads-image]][downloads-url]
[![Build status][travis-image]][travis-url]
[![Test coverage][coveralls-image]][coveralls-url]

Dot case a string.

Supports Unicode (non-ASCII characters) and non-string entities, such as objects with a `toString` property, numbers and booleans. Empty values (`null` and `undefined`) will result in an empty string.

## Installation

```bash
npm install dot-case --save
```

## Usage

```javascript
var dotCase = require('dot-case');

dotCase('string');        //=> "string"
dotCase('camelCase');     //=> "camel.case"
dotCase('sentence case'); //=> "sentence.case"

dotCase('MY STRING', 'tr'); //=> "my.strıng"
```

## License

MIT

[npm-image]: https://img.shields.io/npm/v/dot-case.svg?style=flat
[npm-url]: https://npmjs.org/package/dot-case
[downloads-image]: https://img.shields.io/npm/dm/dot-case.svg?style=flat
[downloads-url]: https://npmjs.org/package/dot-case
[travis-image]: https://img.shields.io/travis/blakeembrey/dot-case.svg?style=flat
[travis-url]: https://travis-ci.org/blakeembrey/dot-case
[coveralls-image]: https://img.shields.io/coveralls/blakeembrey/dot-case.svg?style=flat
[coveralls-url]: https://coveralls.io/r/blakeembrey/dot-case?branch=master
