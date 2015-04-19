# Pascal Case

[![NPM version][npm-image]][npm-url]
[![Build status][travis-image]][travis-url]
[![Test coverage][coveralls-image]][coveralls-url]

Pascal case a string. Explicitly adds a single underscore between groups of numbers to keep readability (E.g. `1.20.5` becomes `1_20_5`, not `1205`).

Supports Unicode (non-ASCII characters) and non-string entities, such as objects with a `toString` property, numbers and booleans. Empty values (`null` and `undefined`) will result in an empty string.

## Installation

```bash
npm install pascal-case --save
```

## Usage

```javascript
var pascalCase = require('pascal-case');

pascalCase('string');        //=> "String"
pascalCase('camelCase');     //=> "CamelCase"
pascalCase('sentence case'); //=> "SentenceCase"

pascalCase('MY STRING', 'tr'); //=> "MyStrıng"
```

## License

MIT

[npm-image]: https://img.shields.io/npm/v/pascal-case.svg?style=flat
[npm-url]: https://npmjs.org/package/pascal-case
[travis-image]: https://img.shields.io/travis/blakeembrey/pascal-case.svg?style=flat
[travis-url]: https://travis-ci.org/blakeembrey/pascal-case
[coveralls-image]: https://img.shields.io/coveralls/blakeembrey/pascal-case.svg?style=flat
[coveralls-url]: https://coveralls.io/r/blakeembrey/pascal-case?branch=master
