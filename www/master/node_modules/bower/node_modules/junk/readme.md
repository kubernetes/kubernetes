# junk [![Build Status](https://travis-ci.org/sindresorhus/junk.svg?branch=master)](https://travis-ci.org/sindresorhus/junk)

> Filter out [OS junk files](test.js) like `.DS_Store` and `Thumbs.db`


## Install

```sh
$ npm install --save junk
```


## Usage

```js
var fs = require('fs');
var junk = require('junk');

fs.readdir('path', function (err, files) {
	console.log(files);
	//=> ['.DS_Store', 'test.jpg']

	console.log(files.filter(junk.not));
	//=> ['test.jpg']
});
```


## API

### junk.is(filename)

Returns true if `filename` matches a junk file.

### junk.not(filename)

Returns true if `filename` doesn't match a junk file.

### junk.re

The regex used for matching.


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
