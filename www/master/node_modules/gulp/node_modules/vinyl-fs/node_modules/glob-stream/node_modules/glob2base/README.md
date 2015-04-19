# glob2base [![NPM version][npm-image]][npm-url] [![Downloads][downloads-image]][npm-url] [![Support us][gittip-image]][gittip-url] [![Build Status][travis-image]][travis-url] [![Coveralls Status][coveralls-image]][coveralls-url]


## Information

<table>
<tr>
<td>Package</td><td>glob2base</td>
</tr>
<tr>
<td>Description</td>
<td>Extracts a base path from a node-glob instance</td>
</tr>
<tr>
<td>Node Version</td>
<td>>= 0.10</td>
</tr>
</table>

## Usage

The module is a function that takes in a node-glob instance and returns a string. Basically it just gives you everything before any globbing/matching happens.

```javascript
var glob2base = require('glob2base');
var glob = require('glob');

// js/
glob2base(new glob.Glob('js/**/*.js'));

// css/test/
glob2base(new glob.Glob('css/test/{a,b}/*.css'));

// pages/whatever/
glob2base(new glob.Glob('pages/whatever/index.html'));
```

## Like what we do?

[gittip-url]: https://www.gittip.com/WeAreFractal/
[gittip-image]: http://img.shields.io/gittip/WeAreFractal.svg

[downloads-image]: http://img.shields.io/npm/dm/glob2base.svg
[npm-url]: https://npmjs.org/package/glob2base
[npm-image]: http://img.shields.io/npm/v/glob2base.svg

[travis-url]: https://travis-ci.org/wearefractal/glob2base
[travis-image]: http://img.shields.io/travis/wearefractal/glob2base.svg

[coveralls-url]: https://coveralls.io/r/wearefractal/glob2base
[coveralls-image]: http://img.shields.io/coveralls/wearefractal/glob2base/master.svg
