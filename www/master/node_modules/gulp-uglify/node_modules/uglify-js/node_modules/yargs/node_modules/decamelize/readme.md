# decamelize [![Build Status](https://travis-ci.org/sindresorhus/decamelize.svg?branch=master)](https://travis-ci.org/sindresorhus/decamelize)

> Convert a camelized string into a lowercased one with a custom separator  
> Example: `unicornRainbow` → `unicorn_rainbow`


## Install

```sh
$ npm install --save decamelize
```


## Usage

```js
var decamelize = require('decamelize');

decamelize('unicornRainbow');
//=> unicorn_rainbow

decamelize('unicornRainbow', '-');
//=> unicorn-rainbow
```


## API

### decamelize(input, [separator])

#### input

*Required*  
Type: `string`

#### separator

Type: `string`  
Default: `_`


## Related

See [`camelcase`](https://github.com/sindresorhus/camelcase) for the inverse.


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
