# figures [![Build Status](https://travis-ci.org/sindresorhus/figures.svg?branch=master)](https://travis-ci.org/sindresorhus/figures)

> Unicode symbols with Windows CMD fallbacks

[![](screenshot.png)](index.js)

[*and more...*](index.js)

Windows CMD only supports a [limited character set](http://en.wikipedia.org/wiki/Code_page_437).


## Install

```sh
$ npm install --save figures
```


## Usage

See the [source](index.js) for supported symbols.

```js
var figures = require('figures');

console.log(figures.tick);
// On real OSes:  ✔︎
// On Windows:    √
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
