# strip-json-comments [![Build Status](https://travis-ci.org/sindresorhus/strip-json-comments.svg?branch=master)](https://travis-ci.org/sindresorhus/strip-json-comments)

> Strip comments from JSON. Lets you use comments in your JSON files!

This is now possible:

```js
{
	// rainbows
	"unicorn": /* ❤ */ "cake"
}
```

It will remove single-line comments `//` and mult-line comments `/**/`.

Also available as a [gulp](https://github.com/sindresorhus/gulp-strip-json-comments)/[grunt](https://github.com/sindresorhus/grunt-strip-json-comments)/[broccoli](https://github.com/sindresorhus/broccoli-strip-json-comments) plugin and a [require hook](https://github.com/uTest/autostrip-json-comments).


*There's already [json-comments](https://npmjs.org/package/json-comments), but it's only for Node.js and uses a naive regex to strip comments which fails on simple cases like `{"a":"//"}`. This module however parses out the comments.*


## Install

```sh
$ npm install --save strip-json-comments
```

```sh
$ bower install --save strip-json-comments
```

```sh
$ component install sindresorhus/strip-json-comments
```


## Usage

```js
var json = '{/*rainbows*/"unicorn":"cake"}';
JSON.parse(stripJsonComments(json));
//=> {unicorn: 'cake'}
```


## API

### stripJsonComments(input)

#### input

Type: `string`

Accepts a string with JSON and returns a string without comments.


## CLI

```sh
$ npm install --global strip-json-comments
```

```sh
$ strip-json-comments --help

strip-json-comments <input-file> > <output-file>
# or
cat <input-file> | strip-json-comments > <output-file>
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
