# vinyl [![NPM version][npm-image]][npm-url] [![Build Status][travis-image]][travis-url] [![Coveralls Status][coveralls-image]][coveralls-url] [![Dependency Status](https://david-dm.org/wearefractal/vinyl.png?theme=shields.io)](https://david-dm.org/wearefractal/vinyl)


## Information

<table>
<tr> 
<td>Package</td><td>vinyl</td>
</tr>
<tr>
<td>Description</td>
<td>A virtual file format</td>
</tr>
<tr>
<td>Node Version</td>
<td>>= 0.9</td>
</tr>
</table>

## File

```javascript
var File = require('vinyl');

var coffeeFile = new File({
  cwd: "/",
  base: "/test/",
  path: "/test/file.coffee"
  contents: new Buffer("test = 123")
});
```

### constructor(options)

#### options.cwd

Type: `String`  
Default: `process.cwd()`

#### options.base

Used for relative pathing. Typically where a glob starts.

Type: `String`  
Default: `options.cwd`

#### options.path

Full path to the file.

Type: `String`  
Default: `null`

#### options.stat

The result of an fs.stat call. See [fs.Stats](http://nodejs.org/api/fs.html#fs_class_fs_stats) for more information.

Type: `fs.Stats`  
Default: `null`

#### options.contents

File contents.

Type: `Buffer, Stream, or null`  
Default: `null`

### isBuffer()

Returns true if file.contents is a Buffer.

### isStream()

Returns true if file.contents is a Stream.

### isNull()

Returns true if file.contents is null.

### clone()

Returns a new File object with all attributes cloned.

### pipe(stream[, opt])

If file.contents is a Buffer, it will write it to the stream.

If file.contents is a Stream, it will pipe it to the stream.

If file.contents is null, it will do nothing.

If opt.end is true, the destination stream will not be ended (same as node core).

Returns the stream.

### inspect()

Returns a pretty String interpretation of the File. Useful for console.log.

### relative

Returns path.relative for the file base and file path.

Example:

```javascript
var file = new File({
  cwd: "/",
  base: "/test/",
  path: "/test/file.coffee"
});

console.log(file.relative); // file.coffee
```

[npm-url]: https://npmjs.org/package/vinyl
[npm-image]: https://badge.fury.io/js/vinyl.png
[travis-url]: https://travis-ci.org/wearefractal/vinyl
[travis-image]: https://travis-ci.org/wearefractal/vinyl.png?branch=master
[coveralls-url]: https://coveralls.io/r/wearefractal/vinyl
[coveralls-image]: https://coveralls.io/repos/wearefractal/vinyl/badge.png
[depstat-url]: https://david-dm.org/wearefractal/vinyl
[depstat-image]: https://david-dm.org/wearefractal/vinyl.png