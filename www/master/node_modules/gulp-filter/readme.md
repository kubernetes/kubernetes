# [gulp](http://gulpjs.com)-filter [![Build Status](https://travis-ci.org/sindresorhus/gulp-filter.svg?branch=master)](https://travis-ci.org/sindresorhus/gulp-filter)

> Filter files in a [vinyl](https://github.com/wearefractal/vinyl) stream

Enables you to work on a subset of the original files by filtering them using globbing. When you're done and want all the original files back you just call the restore method.


## Install

```sh
$ npm install --save-dev gulp-filter
```


## Usage

### Simple

```js
var gulp = require('gulp');
var jscs = require('gulp-jscs');
var gulpFilter = require('gulp-filter');

gulp.task('default', function () {
	// create filter instance inside task function
	var filter = gulpFilter(['*', '!src/vendor']);

	return gulp.src('src/*.js')
		// filter a subset of the files
		.pipe(filter)
		// run them through a plugin
		.pipe(jscs())
		// bring back the previously filtered out files (optional)
		.pipe(filter.restore())
		.pipe(gulp.dest('dist'));
});
```

### Multiple filters

By combining and restoring different filters you can process different sets of files with a single pipeline.

```js
var gulp = require('gulp');
var less = require('gulp-less');
var concat = require('gulp-concat');
var gulpFilter = require('gulp-filter');

gulp.task('default', function () {
	var jsFilter = gulpFilter('**/*.js');
	var lessFilter = gulpFilter('**/*.less');

	return gulp.src('assets/**')
		.pipe(jsFilter)
		.pipe(concat('bundle.js'))
		.pipe(jsFilter.restore())
		.pipe(lessFilter)
		.pipe(less())
		.pipe(lessFilter.restore())
		.pipe(gulp.dest('out/'));
});
```

### Restore as a file source

You may also want to restore filtered files in a different place and use it as a standalone source of files. The `end` option allow you to do so.

```js
var gulp = require('gulp');
var jscs = require('gulp-jscs');
var gulpFilter = require('gulp-filter');

gulp.task('default', function () {
	var filter = gulpFilter(['*', '!src/vendor']);

	return gulp.src('src/*.js')
		// filter a subset of the files
		.pipe(filter)
		// run them through a plugin
		.pipe(jscs())
		.pipe(gulp.dest('dist'));

	// use filtered files as a gulp file source
	filter.restore({end: true})
  		.pipe(gulp.dest('vendor-dist'));
});
```


## API

### filter(pattern, options)

Returns a [transform stream](http://nodejs.org/api/stream.html#stream_class_stream_transform) with a [.restore()](#streamrestore) method.

#### pattern

Type: `string`, `array`, `function`

Accepts a string/array with globbing patterns which are run through [multimatch](https://github.com/sindresorhus/multimatch).

If you supply a function you'll get a [vinyl file object](https://github.com/wearefractal/vinyl#file) as the first argument and you're expected to return true/false whether to include the file:

```js
filter(function (file) {
	return /unicorns/.test(file.path);
});
```

#### options

Type: `object`

Accepts [minimatch options](https://github.com/isaacs/minimatch#options).

*Note:* Set `dot: true` if you need to match files prefixed with a dot (eg. `.gitignore`).


### stream.restore(options)

Brings back the previously filtered out files.

#### options.end

Type: `boolean`  
Default: `false`

Set to `true` if you want restore streams to end when their source stream ends.


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
