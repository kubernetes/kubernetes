# supports-color [![Build Status](https://travis-ci.org/sindresorhus/supports-color.svg?branch=master)](https://travis-ci.org/sindresorhus/supports-color)

> Detect whether a terminal supports color


## Install

```sh
$ npm install --save supports-color
```


## Usage

```js
var supportsColor = require('supports-color');

if (supportsColor) {
	console.log('Terminal supports color');
}
```

It obeys the `--color` and `--no-color` CLI flags.


## CLI

```sh
$ npm install --global supports-color
```

```sh
$ supports-color --help

Usage
  $ supports-color

# Exits with code 0 if color is supported and 1 if not
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
