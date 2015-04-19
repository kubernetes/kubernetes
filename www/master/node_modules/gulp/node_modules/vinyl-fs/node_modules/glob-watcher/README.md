# glob-watcher [![NPM version][npm-image]][npm-url] [![Build Status][travis-image]][travis-url] [![Coveralls Status][coveralls-image]][coveralls-url] [![Dependency Status][david-image]][david-url]

## Information

<table>
<tr> 
<td>Package</td><td>glob-watcher</td>
</tr>
<tr>
<td>Description</td>
<td>Watch globs</td>
</tr>
<tr>
<td>Node Version</td>
<td>>= 0.9</td>
</tr>
</table>

## Usage

```javascript
var watch = require('glob-watcher');

// callback interface
watch(["./*.js", "!./something.js"], function(evt){
  // evt has what file changed and all that jazz
});

// EE interface
var watcher = watch(["./*.js", "!./something.js"]);
watcher.on('change', function(evt) {
  // evt has what file changed and all that jazz
});

// add files after it has been created
watcher.add("./somefolder/somefile.js");
```


[npm-url]: https://npmjs.org/package/glob-watcher
[npm-image]: https://badge.fury.io/js/glob-watcher.png

[travis-url]: https://travis-ci.org/wearefractal/glob-watcher
[travis-image]: https://travis-ci.org/wearefractal/glob-watcher.png?branch=master

[coveralls-url]: https://coveralls.io/r/wearefractal/glob-watcher
[coveralls-image]: https://coveralls.io/repos/wearefractal/glob-watcher/badge.png

[depstat-url]: https://david-dm.org/wearefractal/glob-watcher
[depstat-image]: https://david-dm.org/wearefractal/glob-watcher.png

[david-url]: https://david-dm.org/wearefractal/glob-watcher
[david-image]: https://david-dm.org/wearefractal/glob-watcher.png?theme=shields.io
