# multimatch [![Build Status](https://travis-ci.org/sindresorhus/multimatch.svg?branch=master)](https://travis-ci.org/sindresorhus/multimatch)

> Extends [`minimatch.match()`](https://github.com/isaacs/minimatch#minimatchmatchlist-pattern-options) with support for multiple patterns


#### Comparison

Minimatch:

```js
minimatch.match(['unicorn', 'cake', 'rainbows'], '*corn');
```

Multimatch:

```js
multimatch(['unicorn', 'cake', 'rainbows'], ['*corn', 'rain*']);
```


## Install

```sh

$ npm install --save multimatch
```


## Usage

```js
var multimatch = require('multimatch');

multimatch(['unicorn', 'cake', 'rainbows'], ['*', '!cake']);
//=> ['unicorn', 'rainbows']
```

See the [tests](https://github.com/sindresorhus/multimatch/blob/master/test.js) for more usage examples and expected matches.


## API

Same as [`minimatch.match()`](https://github.com/isaacs/minimatch#minimatchmatchlist-pattern-options) except for `pattern` also accepting an array.


## License

MIT © [Sindre Sorhus](http://sindresorhus.com), [Jon Schlinkert](https://github.com/jonschlinkert)
