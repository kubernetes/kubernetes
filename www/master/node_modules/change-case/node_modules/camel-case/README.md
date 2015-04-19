# Camel Case

[![NPM version][npm-image]][npm-url]
[![Build status][travis-image]][travis-url]
[![Test coverage][coveralls-image]][coveralls-url]

Camel case a string. Explicitly adds a single underscore between groups of numbers to keep readability (E.g. `1.20.5` becomes `1_20_5`, not `1205`).

Supports Unicode (non-ASCII characters) and non-string entities, such as objects with a `toString` property, numbers and booleans. Empty values (`null` and `undefined`) will result in an empty string.

## Installation

### Node

```sh
npm install camel-case --save
```

## Usage

```javascript
var camelCase = require('camel-case');

camelCase('string');         //=> "string"
camelCase('dot.case');       //=> "dotCase"
camelCase('PascalCase');     //=> "pascalCase"
camelCase('version 1.2.10'); //=> "version1_2_10"

camelCase('STRING 1.2', 'tr'); //=> "strıng1_2"
```

## License

MIT

[npm-image]: https://img.shields.io/npm/v/camel-case.svg?style=flat
[npm-url]: https://npmjs.org/package/camel-case
[travis-image]: https://img.shields.io/travis/blakeembrey/camel-case.svg?style=flat
[travis-url]: https://travis-ci.org/blakeembrey/camel-case
[coveralls-image]: https://img.shields.io/coveralls/blakeembrey/camel-case.svg?style=flat
[coveralls-url]: https://coveralls.io/r/blakeembrey/camel-case?branch=master
