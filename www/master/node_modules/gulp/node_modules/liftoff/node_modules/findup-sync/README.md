# findup-sync [![Build Status](https://secure.travis-ci.org/cowboy/node-findup-sync.png?branch=master)](http://travis-ci.org/cowboy/node-findup-sync)

Find the first file matching a given pattern in the current directory or the nearest ancestor directory.

## Getting Started
Install the module with: `npm install findup-sync`

```js
var findup = require('findup-sync');

// Start looking in the CWD.
var filepath1 = findup('{a,b}*.txt');

// Start looking somewhere else, and ignore case (probably a good idea).
var filepath2 = findup('{a,b}*.txt', {cwd: '/some/path', nocase: true});
```

## Usage

```js
findup(patternOrPatterns [, minimatchOptions])
```

### patternOrPatterns
Type: `String` or `Array`  
Default: none

One or more wildcard glob patterns. Or just filenames.

### minimatchOptions
Type: `Object`  
Default: `{}`

Options to be passed to [minimatch](https://github.com/isaacs/minimatch).

Note that if you want to start in a different directory than the current working directory, specify a `cwd` property here.

## Contributing
In lieu of a formal styleguide, take care to maintain the existing coding style. Add unit tests for any new or changed functionality. Lint and test your code using [Grunt](http://gruntjs.com/).

## Release History
2014-12-17 - v0.2.1 - updated to glob 4.3.  
2014-12-16 - v0.2.0 - Removed lodash, updated to glob 4.x.  
2014-03-14 - v0.1.3 - Updated dependencies.  
2013-03-08 - v0.1.2 - Updated dependencies. Fixed a Node 0.9.x bug. Updated unit tests to work cross-platform.  
2012-11-15 - v0.1.1 - Now works without an options object.  
2012-11-01 - v0.1.0 - Initial release.
