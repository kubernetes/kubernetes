# del [![Build Status](https://travis-ci.org/sindresorhus/del.svg?branch=master)](https://travis-ci.org/sindresorhus/del)

> Delete files/folders using [globs](https://github.com/isaacs/minimatch#usage)

Pretty much [rimraf](https://github.com/isaacs/rimraf) with support for multiple files and globbing.  
It also protects you against deleting the current working directory and above.


## Install

```sh
$ npm install --save del
```


## Usage

```js
var del = require('del');

del(['tmp/*.js', '!tmp/unicorn.js'], function (err, deletedFiles) {
	console.log('Files deleted:', deletedFiles.join(', '));
});
```


## API

### del(patterns, [options], callback)
### del.sync(patterns, [options])

The async method gets an array of deleted files as the second argument in the callback, while the sync method returns the array.

#### patterns

**Required**  
Type: `string`, `array`

See supported minimatch [patterns](https://github.com/isaacs/minimatch#usage).

- [Pattern examples with expected matches](https://github.com/sindresorhus/multimatch/blob/master/test.js)
- [Quick globbing pattern overview](https://github.com/sindresorhus/multimatch#globbing-patterns)

#### options

Type: `object`

See the node-glob [options](https://github.com/isaacs/node-glob#options).

#### options.force

Type: `boolean`  
Default: `false`

Allow deleting the current working directory and files/folders outside it.


## CLI

See [trash](https://github.com/sindresorhus/trash).


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
