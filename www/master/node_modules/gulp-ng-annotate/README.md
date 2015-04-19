# [gulp](http://gulpjs.com)-ng-annotate [![Build Status](https://travis-ci.org/Kagami/gulp-ng-annotate.svg?branch=master)](https://travis-ci.org/Kagami/gulp-ng-annotate)

> Add angularjs dependency injection annotations with [ng-annotate](https://github.com/olov/ng-annotate)

## Install

```bash
$ npm install --save-dev gulp-ng-annotate
```

## Usage

```js
var gulp = require('gulp');
var ngAnnotate = require('gulp-ng-annotate');

gulp.task('default', function () {
	return gulp.src('src/app.js')
		.pipe(ngAnnotate())
		.pipe(gulp.dest('dist'));
});
```

## Options

You can pass any of the [ng-annotate options](https://github.com/olov/ng-annotate#installation-and-usage) as an object:
```js
{
	remove: true,
	add: true,
	single_quotes: true
}
```

If no options provided, plugin will be executed with `{add: true}`

## License

gulp-ng-annotate - Add angularjs dependency injection annotations with ng-annotate

Written in 2014 by Kagami Hiiragi <kagami@genshiken.org>

To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
