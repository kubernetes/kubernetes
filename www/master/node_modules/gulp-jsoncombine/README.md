# gulp-jsoncombine
[![NPM version][npm-image]][npm-url] [![Build Status][travis-image]][travis-url]  [![Coverage Status][coveralls-image]][coveralls-url] [![Dependency Status][depstat-image]][depstat-url]

> jsoncombine plugin for [gulp](https://github.com/wearefractal/gulp)

## Usage

First, install `gulp-jsoncombine` as a development dependency:

```shell
npm install --save-dev gulp-jsoncombine
```

Then, add it to your `gulpfile.js`:

** This plugin will collect all the json files provided to it, parse them, put them in a dictionary where the keys of that dictionary are the filenames (sans the '.json' postfix) and pass that to a processor function. That function decides how that output should look in the resulting file. **

```javascript
var jsoncombine = require("gulp-jsoncombine");

gulp.src("./src/*.json")
	.pipe(jsoncombine("result.js",function(data){...}))
	.pipe(gulp.dest("./dist"));
```

## API

### jsoncombine(fileName, processor)

#### fileName
Type: `String`  

The output filename

#### processor
Type: `Function`  

The function that will be called with the dictionary containing all the data from the processes JSON files, where the keys of the dictionary, would be the names of the files (sans the '.json' postfix).

The function should return a new `Buffer` that would be writter to the output file.


## License

[MIT License](http://en.wikipedia.org/wiki/MIT_License)

[npm-url]: https://npmjs.org/package/gulp-jsoncombine
[npm-image]: https://badge.fury.io/js/gulp-jsoncombine.png

[travis-url]: http://travis-ci.org/reflog/gulp-jsoncombine
[travis-image]: https://secure.travis-ci.org/reflog/gulp-jsoncombine.png?branch=master

[coveralls-url]: https://coveralls.io/r/reflog/gulp-jsoncombine
[coveralls-image]: https://coveralls.io/repos/reflog/gulp-jsoncombine/badge.png

[depstat-url]: https://david-dm.org/reflog/gulp-jsoncombine
[depstat-image]: https://david-dm.org/reflog/gulp-jsoncombine.png
