# array-differ [![Build Status](https://travis-ci.org/sindresorhus/array-differ.svg?branch=master)](https://travis-ci.org/sindresorhus/array-differ)

> Create an array with values that are present in the first input array but not additional ones


## Install

```sh
$ npm install --save array-differ
```


## Usage

```js
var arrayDiffer = require('array-differ');

arrayDiffer([2, 3, 4], [3, 50]);
//=> [2, 4]
```

## API

### arrayDiffer(input, values, [values, ...])

Returns the new array.

#### input

Type: `array`

#### values

Type: `array`

Arrays of values to exclude.


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
