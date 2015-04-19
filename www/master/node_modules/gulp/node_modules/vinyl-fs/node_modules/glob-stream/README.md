# glob-stream [![NPM version][npm-image]][npm-url] [![Build Status][travis-image]][travis-url] [![Coveralls Status][coveralls-image]][coveralls-url] [![Dependency Status][david-image]][david-url]


## Information

<table>
<tr> 
<td>Package</td><td>glob-stream</td>
</tr>
<tr>
<td>Description</td>
<td>File system globs as a stream</td>
</tr>
<tr>
<td>Node Version</td>
<td>>= 0.9</td>
</tr>
</table>

This is a simple wrapper around node-glob to make it streamy.

## Usage

```javascript
var gs = require('glob-stream');

var stream = gs.create("./files/**/*.coffee", {options});

stream.on('data', function(file){
  // file has path, base, and cwd attrs
});
```

You can pass any combination of globs. One caveat is that you can not only pass a glob negation, you must give it at least one positive glob so it knows where to start. All given must match for the file to be returned.

### Options

- cwd
  - Default is `process.cwd()`
- base
  - Default is everything before a glob starts (see [glob2base](https://github.com/wearefractal/glob2base))
- cwdbase
  - Default is `false`
  - When true it is the same as saying opt.base = opt.cwd

This argument is passed directly to [node-glob](https://github.com/isaacs/node-glob) so check there for more options

#### Glob

```javascript
var stream = gs.create(["./**/*.js", "!./node_modules/**/*.*"]);
```

[npm-url]: https://npmjs.org/package/glob-stream
[npm-image]: https://badge.fury.io/js/glob-stream.png

[travis-url]: https://travis-ci.org/wearefractal/glob-stream
[travis-image]: https://travis-ci.org/wearefractal/glob-stream.png?branch=master

[coveralls-url]: https://coveralls.io/r/wearefractal/glob-stream
[coveralls-image]: https://coveralls.io/repos/wearefractal/glob-stream/badge.png

[depstat-url]: https://david-dm.org/wearefractal/glob-stream
[depstat-image]: https://david-dm.org/wearefractal/glob-stream.png

[david-url]: https://david-dm.org/wearefractal/glob-stream
[david-image]: https://david-dm.org/wearefractal/glob-stream.png?theme=shields.io
