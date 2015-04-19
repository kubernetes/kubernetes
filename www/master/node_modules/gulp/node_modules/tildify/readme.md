# tildify [![Build Status](https://travis-ci.org/sindresorhus/tildify.svg?branch=master)](https://travis-ci.org/sindresorhus/tildify)

> Convert an absolute path to a tilde path: `/Users/sindresorhus/dev` => `~/dev`


## Install

```sh
$ npm install --save tildify
```


## Usage

```js
var tildify = require('tildify');

tildify('/Users/sindresorhus/dev');
//=> ~/dev
```


## Related

See [untildify](https://github.com/sindresorhus/untildify) for the inverse.


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
