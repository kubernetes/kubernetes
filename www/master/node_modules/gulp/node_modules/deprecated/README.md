# deprecated [![NPM version][npm-image]][npm-url] [![Build Status][travis-image]][travis-url] [![Coveralls Status][coveralls-image]][coveralls-url] [![Dependency Status][david-image]][david-url]


## Information

<table>
<tr> 
<td>Package</td><td>deprecated</td>
</tr>
<tr>
<td>Description</td>
<td>Tool for deprecating things</td>
</tr>
<tr>
<td>Node Version</td>
<td>>= 0.9</td>
</tr>
</table>

## Usage

```javascript
var oldfn = function(a,b) {
  return a+b;
};

// returns a new wrapper function that logs the deprecated function once
var somefn = deprecated('dont use this anymore', console.log, oldfn);

var someobj = {};

// set up a getter/set for field that logs deprecated message once
deprecated('dont use this anymore', console.log, someobj, 'a', 123);

console.log(someobj.a); // 123
```

[npm-url]: https://npmjs.org/package/deprecated
[npm-image]: https://badge.fury.io/js/deprecated.png

[travis-url]: https://travis-ci.org/wearefractal/deprecated
[travis-image]: https://travis-ci.org/wearefractal/deprecated.png?branch=master

[coveralls-url]: https://coveralls.io/r/wearefractal/deprecated
[coveralls-image]: https://coveralls.io/repos/wearefractal/deprecated/badge.png

[depstat-url]: https://david-dm.org/wearefractal/deprecated
[depstat-image]: https://david-dm.org/wearefractal/deprecated.png

[david-url]: https://david-dm.org/wearefractal/deprecated
[david-image]: https://david-dm.org/wearefractal/deprecated.png?theme=shields.io