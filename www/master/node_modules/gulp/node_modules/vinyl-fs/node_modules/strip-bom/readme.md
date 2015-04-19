# strip-bom [![Build Status](https://travis-ci.org/sindresorhus/strip-bom.svg?branch=master)](https://travis-ci.org/sindresorhus/strip-bom)

> Strip UTF-8 [byte order mark](http://en.wikipedia.org/wiki/Byte_order_mark#UTF-8) (BOM) from a string/buffer/stream

From Wikipedia:

> The Unicode Standard permits the BOM in UTF-8, but does not require nor recommend its use. Byte order has no meaning in UTF-8.


## Usage

```sh
$ npm install --save strip-bom
```

```js
var fs = require('fs');
var stripBom = require('strip-bom');

stripBom('\ufeffUnicorn');
//=> Unicorn

stripBom(fs.readFileSync('unicorn.txt'));
//=> Unicorn
```

Or as a [Transform stream](http://nodejs.org/api/stream.html#stream_class_stream_transform):

```js
var fs = require('fs');
var stripBom = require('strip-bom');

fs.createReadStream('unicorn.txt')
	.pipe(stripBom.stream())
	.pipe(fs.createWriteStream('unicorn.txt'));
```


## CLI

```sh
$ npm install --global strip-bom
```

```
$ strip-bom --help

  Usage
    strip-bom <file> > <new-file>
    cat <file> | strip-bom > <new-file>

  Example
    strip-bom unicorn.txt > unicorn-without-bom.txt
```


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
