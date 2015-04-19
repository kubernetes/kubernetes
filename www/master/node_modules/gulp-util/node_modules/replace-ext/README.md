# replace-ext [![NPM version][npm-image]][npm-url] [![Build Status][travis-image]][travis-url] [![Coveralls Status][coveralls-image]][coveralls-url] [![Dependency Status][david-image]][david-url]


## Information

<table>
<tr> 
<td>Package</td><td>replace-ext</td>
</tr>
<tr>
<td>Description</td>
<td>Replaces a file extension with another one</td>
</tr>
<tr>
<td>Node Version</td>
<td>>= 0.4</td>
</tr>
</table>

## Usage

```javascript
var replaceExt = require('replace-ext');

var path = '/some/dir/file.js';
var npath = replaceExt(path, '.coffee');

console.log(npath); // /some/dir/file.coffee
```

[npm-url]: https://npmjs.org/package/replace-ext
[npm-image]: https://badge.fury.io/js/replace-ext.png

[travis-url]: https://travis-ci.org/wearefractal/replace-ext
[travis-image]: https://travis-ci.org/wearefractal/replace-ext.png?branch=master

[coveralls-url]: https://coveralls.io/r/wearefractal/replace-ext
[coveralls-image]: https://coveralls.io/repos/wearefractal/replace-ext/badge.png

[depstat-url]: https://david-dm.org/wearefractal/replace-ext
[depstat-image]: https://david-dm.org/wearefractal/replace-ext.png

[david-url]: https://david-dm.org/wearefractal/replace-ext
[david-image]: https://david-dm.org/wearefractal/replace-ext.png?theme=shields.io