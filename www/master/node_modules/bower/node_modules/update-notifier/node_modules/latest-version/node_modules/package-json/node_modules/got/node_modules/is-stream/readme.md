# is-stream [![Build Status](https://travis-ci.org/sindresorhus/is-stream.svg?branch=master)](https://travis-ci.org/sindresorhus/is-stream)

> Check if something is a [Node.js stream](http://nodejs.org/api/stream.html)


## Install

```
$ npm install --save is-stream
```


## Usage

```js
var fs = require('fs');
var isStream = require('is-stream');

isStream(fs.createReadStream('unicorn.png'));
//=> true

isStream({});
//=> false
```


## API

### isStream(stream)

#### isStream.writable(stream)

#### isStream.readable(stream)

#### isStream.duplex(stream)



## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
