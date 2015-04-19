# camelcase-keys [![Build Status](https://travis-ci.org/sindresorhus/camelcase-keys.svg?branch=master)](https://travis-ci.org/sindresorhus/camelcase-keys)

> Convert object keys to camelCase using [`camelcase`](https://github.com/sindresorhus/camelcase)


## Install

```sh
$ npm install --save camelcase-keys
```


## Usage

```js
var camelcaseKeys = require('camelcase-keys');

camelcaseKeys({'foo-bar': true});
//=> {fooBar: true}


var argv = require('minimist')(process.argv.slice(2));
//=> {_: [], 'foo-bar': true}

camelcaseKeys(argv);
//=> {_: [], fooBar: true}
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
