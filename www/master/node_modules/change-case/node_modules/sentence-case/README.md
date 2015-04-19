# Sentence Case

[![NPM version][npm-image]][npm-url]
[![NPM downloads][downloads-image]][downloads-url]
[![Build status][travis-image]][travis-url]
[![Test coverage][coveralls-image]][coveralls-url]

Sentence case a string. Optional locale and replacement character supported.

Supports Unicode (non-ASCII characters) and non-string entities, such as objects with a `toString` property, numbers and booleans. Empty values (`null` and `undefined`) will result in an empty string.

## Installation

```bash
npm install sentence-case --save
```

## Usage

```javascript
var sentenceCase = require('sentence-case')

sentenceCase(null)              //=> ""
sentenceCase('string')          //=> "string"
sentenceCase('dot.case')        //=> "dot case"
sentenceCase('camelCase')       //=> "camel case"
sentenceCase('Beyoncé Knowles') //=> "beyoncé knowles"

sentenceCase('A STRING', 'tr') //=> "a strıng"

sentenceCase('HELLO WORLD!', null, '_') //=> "hello_world"
```

## License

MIT

[npm-image]: https://img.shields.io/npm/v/sentence-case.svg?style=flat
[npm-url]: https://npmjs.org/package/sentence-case
[downloads-image]: https://img.shields.io/npm/dm/sentence-case.svg?style=flat
[downloads-url]: https://npmjs.org/package/sentence-case
[travis-image]: https://img.shields.io/travis/blakeembrey/sentence-case.svg?style=flat
[travis-url]: https://travis-ci.org/blakeembrey/sentence-case
[coveralls-image]: https://img.shields.io/coveralls/blakeembrey/sentence-case.svg?style=flat
[coveralls-url]: https://coveralls.io/r/blakeembrey/sentence-case?branch=master
