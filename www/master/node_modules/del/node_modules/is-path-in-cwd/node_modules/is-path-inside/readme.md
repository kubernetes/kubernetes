# is-path-inside [![Build Status](https://travis-ci.org/sindresorhus/is-path-inside.svg?branch=master)](https://travis-ci.org/sindresorhus/is-path-inside)

> Check if a path is inside another path


## Install

```sh
$ npm install --save is-path-inside
```


## Usage

```js
var isPathInside = require('is-path-inside');

isPathInside('a/b', 'a/b/c');
//=> true

isPathInside('x/y', 'a/b/c');
//=> false

isPathInside('a/b/c', 'a/b/c');
//=> false
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
