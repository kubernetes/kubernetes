# Change Case

[![NPM version][npm-image]][npm-url]
[![NPM downloads][downloads-image]][downloads-url]
[![Build status][travis-image]][travis-url]
[![Test coverage][coveralls-image]][coveralls-url]

Convert strings between `camelCase`, `PascalCase`, `Title Case`, `snake_case`, `lowercase`, `UPPERCASE`, `CONSTANT_CASE` and more.

All methods support Unicode (non-ASCII characters) and non-string entities, such as objects with a `toString` property, numbers and booleans. Empty values (`null` and `undefined`) will result in an empty string.

**Every method is also available on npm as an individual package.**

## Installation

```
npm install change-case --save
```

## Usage

```js
var changeCase = require('change-case')
//=> { isUpperCase: [Function], camelCase: [Function], ... }
```

**Available methods** (short-hand shown below, long-hand available in examples):

* `isUpper`
* `isLower`
* `upper`
* `ucFirst`
* `lcFirst`
* `lower`
* `sentence`
* `title`
* `camel`
* `pascal`
* `snake`
* `param`
* `dot`
* `path`
* `constant`
* `swap`

All methods accept two arguments, the string to change case and an optional locale.

### [isUpperCase](https://github.com/blakeembrey/is-upper-case)

[![NPM version](https://img.shields.io/npm/v/is-upper-case.svg?style=flat)](https://npmjs.org/package/is-upper-case)
[![NPM downloads](https://img.shields.io/npm/dm/is-upper-case.svg?style=flat)](https://npmjs.org/package/is-upper-case)
[![Build status](https://img.shields.io/travis/blakeembrey/is-upper-case.svg?style=flat)](https://travis-ci.org/blakeembrey/is-upper-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/is-upper-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/is-upper-case?branch=master)

Return a boolean indicating whether the string is upper cased.

```js
changeCase.isUpperCase('test string')
//=> false
```

### [isLowerCase](https://github.com/blakeembrey/is-lower-case)

[![NPM version](https://img.shields.io/npm/v/is-lower-case.svg?style=flat)](https://npmjs.org/package/is-lower-case)
[![NPM downloads](https://img.shields.io/npm/dm/is-lower-case.svg?style=flat)](https://npmjs.org/package/is-lower-case)
[![Build status](https://img.shields.io/travis/blakeembrey/is-lower-case.svg?style=flat)](https://travis-ci.org/blakeembrey/is-lower-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/is-lower-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/is-lower-case?branch=master)

Return a boolean indicating whether the string is lower cased.

```js
changeCase.isLowerCase('test string')
//=> true
```

### [upperCase](https://github.com/blakeembrey/upper-case)

[![NPM version](https://img.shields.io/npm/v/upper-case.svg?style=flat)](https://npmjs.org/package/upper-case)
[![NPM downloads](https://img.shields.io/npm/dm/upper-case.svg?style=flat)](https://npmjs.org/package/upper-case)
[![Build status](https://img.shields.io/travis/blakeembrey/upper-case.svg?style=flat)](https://travci.org/blakeembrey/upper-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/upper-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/is-upper-case?branch=master)

Return the string in upper case.

```js
changeCase.upperCase('test string')
//=> "TEST STRING"
```

### [upperCaseFirst](https://github.com/blakeembrey/upper-case-first)

[![NPM version](https://img.shields.io/npm/v/upper-case-first.svg?style=flat)](https://npmjs.org/package/upper-case-first)
[![NPM downloads](https://img.shields.io/npm/dm/upper-case-first.svg?style=flat)](https://npmjs.org/package/upper-case-first)
[![Build status](https://img.shields.io/travis/blakeembrey/upper-case-first.svg?style=flat)](https://travis-ci.org/blakeembrey/upper-case-first)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/upper-case-first.svg?style=flat)](https://coveralls.io/r/blakeembrey/is-upper-case?branch=master)

Return the string with the first character upper cased.

```js
changeCase.upperCaseFirst('test')
//=> "Test"
```

### [lowerCaseFirst](https://github.com/blakeembrey/lower-case-first)

[![NPM version](https://img.shields.io/npm/v/lower-case-first.svg?style=flat)](https://npmjs.org/package/lower-case-first)
[![NPM downloads](https://img.shields.io/npm/dm/lower-case-first.svg?style=flat)](https://npmjs.org/package/lower-case-first)
[![Build status](https://img.shields.io/travis/blakeembrey/lower-case-first.svg?style=flat)](https://travis-ci.org/blakeembrey/lower-case-first)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/lower-case-first.svg?style=flat)](https://coveralls.io/r/blakeembrey/lower-case-first?branch=master)

Return the string with the first character lower cased.

```js
changeCase.lowerCaseFirst('TEST')
//=> "tEST"
```

### [lowerCase](https://github.com/blakeembrey/lower-case)

[![NPM version](https://img.shields.io/npm/v/lower-case.svg?style=flat)](https://npmjs.org/package/lower-case)
[![NPM downloads](https://img.shields.io/npm/dm/lower-case.svg?style=flat)](https://npmjs.org/package/lower-case)
[![Build status](https://img.shields.io/travis/blakeembrey/lower-case.svg?style=flat)](https://travis-ci.org/blakeembrey/lower-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/lower-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/lower-case?branch=master)

Return the string in lower case.

```js
changeCase.lowerCase('TEST STRING')
//=> "test string"
```

### [sentenceCase](https://github.com/blakeembrey/sentence-case)

[![NPM version](https://img.shields.io/npm/v/sentence-case.svg?style=flat)](https://npmjs.org/package/sentence-case)
[![NPM downloads](https://img.shields.io/npm/dm/sentence-case.svg?style=flat)](https://npmjs.org/package/sentence-case)
[![Build status](https://img.shields.io/travis/blakeembrey/sentence-case.svg?style=flat)](https://travis-ci.org/blakeembrey/sentence-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/sentence-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/sentence-case?branch=master)

Return as a lower case, space separated string.

```js
changeCase.sentenceCase('testString')
//=> "test string"
```

### [titleCase](https://github.com/blakeembrey/title-case)

[![NPM version](https://img.shields.io/npm/v/title-case.svg?style=flat)](https://npmjs.org/package/title-case)
[![NPM downloads](https://img.shields.io/npm/dm/title-case.svg?style=flat)](https://npmjs.org/package/title-case)
[![Build status](https://img.shields.io/travis/blakeembrey/title-case.svg?style=flat)](https://travis-ci.org/blakeembrey/title-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/title-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/title-case?branch=master)

Return as a space separated string with the first character of every word upper cased.

```js
changeCase.titleCase('a simple test')
//=> "A Simple Test"
```

### [camelCase](https://github.com/blakeembrey/camel-case)

[![NPM version](https://img.shields.io/npm/v/camel-case.svg?style=flat)](https://npmjs.org/package/camel-case)
[![NPM downloads](https://img.shields.io/npm/dm/camel-case.svg?style=flat)](https://npmjs.org/package/camel-case)
[![Build status](https://img.shields.io/travis/blakeembrey/camel-case.svg?style=flat)](https://travis-ci.org/blakeembrey/camel-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/camel-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/camel-case?branch=master)

Return as a string with the separators denoted by having the next letter capitalized.

```js
changeCase.camelCase('test string')
//=> "testString"
```

### [pascalCase](https://github.com/blakeembrey/pascal-case)

[![NPM version](https://img.shields.io/npm/v/pascal-case.svg?style=flat)](https://npmjs.org/package/pascal-case)
[![NPM downloads](https://img.shields.io/npm/dm/pascal-case.svg?style=flat)](https://npmjs.org/package/pascal-case)
[![Build status](https://img.shields.io/travis/blakeembrey/pascal-case.svg?style=flat)](https://travis-ci.org/blakeembrey/pascal-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/pascal-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/pascal-case?branch=master)

Return as a string denoted in the same fashion as `camelCase`, but with the first letter also capitalized.

```js
changeCase.pascalCase('test string')
//=> "TestString"
```

### [snakeCase](https://github.com/blakeembrey/snake-case)

[![NPM version](https://img.shields.io/npm/v/snake-case.svg?style=flat)](https://npmjs.org/package/snake-case)
[![NPM downloads](https://img.shields.io/npm/dm/snake-case.svg?style=flat)](https://npmjs.org/package/snake-case)
[![Build status](https://img.shields.io/travis/blakeembrey/snake-case.svg?style=flat)](https://travis-ci.org/blakeembrey/snake-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/snake-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/snake-case?branch=master)

Return as a lower case, underscore separated string.

```js
changeCase.snakeCase('test string')
//=> "test_string"
```

### [paramCase](https://github.com/blakeembrey/param-case)

[![NPM version](https://img.shields.io/npm/v/param-case.svg?style=flat)](https://npmjs.org/package/param-case)
[![NPM downloads](https://img.shields.io/npm/dm/param-case.svg?style=flat)](https://npmjs.org/package/param-case)
[![Build status](https://img.shields.io/travis/blakeembrey/param-case.svg?style=flat)](https://travis-ci.org/blakeembrey/param-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/param-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/param-case?branch=master)

Return as a lower case, dash separated string.

```js
changeCase.paramCase('test string')
//=> "test-string"
```

### [dotCase](https://github.com/blakeembrey/dot-case)

[![NPM version](https://img.shields.io/npm/v/dot-case.svg?style=flat)](https://npmjs.org/package/dot-case)
[![NPM downloads](https://img.shields.io/npm/dm/dot-case.svg?style=flat)](https://npmjs.org/package/dot-case)
[![Build status](https://img.shields.io/travis/blakeembrey/dot-case.svg?style=flat)](https://travis-ci.org/blakeembrey/dot-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/dot-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/dot-case?branch=master)

Return as a lower case, period separated string.

```js
changeCase.dotCase('test string')
//=> "test.string"
```

### [pathCase](https://github.com/blakeembrey/path-case)

[![NPM version](https://img.shields.io/npm/v/path-case.svg?style=flat)](https://npmjs.org/package/path-case)
[![NPM downloads](https://img.shields.io/npm/dm/path-case.svg?style=flat)](https://npmjs.org/package/path-case)
[![Build status](https://img.shields.io/travis/blakeembrey/path-case.svg?style=flat)](https://travis-ci.org/blakeembrey/path-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/path-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/path-case?branch=master)

Return as a lower case, slash separated string.

```js
changeCase.pathCase('test string')
//=> "test/string"
```

### [constantCase](https://github.com/blakeembrey/constant-case)

[![NPM version](https://img.shields.io/npm/v/constant-case.svg?style=flat)](https://npmjs.org/package/constant-case)
[![NPM downloads](https://img.shields.io/npm/dm/constant-case.svg?style=flat)](https://npmjs.org/package/constant-case)
[![Build status](https://img.shields.io/travis/blakeembrey/constant-case.svg?style=flat)](https://travis-ci.org/blakeembrey/constant-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/constant-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/constant-case?branch=master)

Return as an upper case, underscore separated string.

```js
changeCase.constantCase('test string')
//=> "TEST_STRING"
```

### [swapCase](https://github.com/blakeembrey/swap-case)

[![NPM version](https://img.shields.io/npm/v/swap-case.svg?style=flat)](https://npmjs.org/package/swap-case)
[![NPM downloads](https://img.shields.io/npm/dm/swap-case.svg?style=flat)](https://npmjs.org/package/swap-case)
[![Build status](https://img.shields.io/travis/blakeembrey/swap-case.svg?style=flat)](https://travis-ci.org/blakeembrey/swap-case)
[![Test coverage](https://img.shields.io/coveralls/blakeembrey/swap-case.svg?style=flat)](https://coveralls.io/r/blakeembrey/swap-case?branch=master)

Return as a string with every character case reversed.

```js
changeCase.swapCase('Test String')
//=> "tEST sTRING"
```

## Related

Also available on [Meteor](https://github.com/Konecty/change-case)!

## License

MIT

[npm-image]: https://img.shields.io/npm/v/change-case.svg?style=flat
[npm-url]: https://npmjs.org/package/change-case
[downloads-image]: https://img.shields.io/npm/dm/change-case.svg?style=flat
[downloads-url]: https://npmjs.org/package/change-case
[travis-image]: https://img.shields.io/travis/blakeembrey/change-case.svg?style=flat
[travis-url]: https://travis-ci.org/blakeembrey/change-case
[coveralls-image]: https://img.shields.io/coveralls/blakeembrey/change-case.svg?style=flat
[coveralls-url]: https://coveralls.io/r/blakeembrey/change-case?branch=master
