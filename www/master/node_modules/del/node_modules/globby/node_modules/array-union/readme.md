# array-union [![Build Status](https://travis-ci.org/sindresorhus/array-union.svg?branch=master)](https://travis-ci.org/sindresorhus/array-union)

> Create an array of unique values, in order, from the input arrays


## Install

```sh
$ npm install --save array-union
```


## Usage

```js
var arrayUnion = require('array-union');

arrayUnion([1, 1, 2, 3], [2, 3]);
//=> [1, 2, 3]

arrayUnion(['foo', 'foo', 'bar'], ['foo']);
//=> ['foo', 'bar']
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
