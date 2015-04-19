# Snake Case

[![NPM version][npm-image]][npm-url]
[![NPM downloads][downloads-image]][downloads-url]
[![Build status][travis-image]][travis-url]
[![Test coverage][coveralls-image]][coveralls-url]

Snake case a string.

Supports Unicode (non-ASCII characters) and non-string entities, such as objects with a `toString` property, numbers and booleans. Empty values (`null` and `undefined`) will result in an empty string.

## Installation

```bash
npm install snake-case --save
```

## Usage

```javascript
var snakeCase = require('snake-case');

snakeCase('string');        //=> "string"
snakeCase('camelCase');     //=> "camel_case"
snakeCase('sentence case'); //=> "sentence_case"

snakeCase('MY STRING', 'tr'); //=> "my_strıng"
```

## License

MIT

[npm-image]: https://img.shields.io/npm/v/snake-case.svg?style=flat
[npm-url]: https://npmjs.org/package/snake-case
[downloads-image]: https://img.shields.io/npm/dm/snake-case.svg?style=flat
[downloads-url]: https://npmjs.org/package/snake-case
[travis-image]: https://img.shields.io/travis/blakeembrey/snake-case.svg?style=flat
[travis-url]: https://travis-ci.org/blakeembrey/snake-case
[coveralls-image]: https://img.shields.io/coveralls/blakeembrey/snake-case.svg?style=flat
[coveralls-url]: https://coveralls.io/r/blakeembrey/snake-case?branch=master
