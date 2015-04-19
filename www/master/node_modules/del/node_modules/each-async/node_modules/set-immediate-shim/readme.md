# set-immediate-shim [![Build Status](https://travis-ci.org/sindresorhus/set-immediate-shim.svg?branch=master)](https://travis-ci.org/sindresorhus/set-immediate-shim)

> Simple [`setImmediate`](https://developer.mozilla.org/en-US/docs/Web/API/Window.setImmediate) shim


## Install

```
$ npm install --save set-immediate-shim
```


## Usage

```js
var setImmediateShim = require('set-immediate-shim');

setImmediateShim(function () {
	console.log('2');
});

console.log('1');

//=> 1
//=> 2
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
