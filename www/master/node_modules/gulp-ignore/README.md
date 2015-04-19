gulp-ignore ![status](https://secure.travis-ci.org/robrich/gulp-ignore.png?branch=master)
===========

Include or exclude [gulp](https://github.com/gulpjs/gulp) files from the stream based on a condition

Usage

1: Exclude things from the stream

**Exclude things from entering the stream**

![][glob]

```javascript
var uglify = require('gulp-uglify');

gulp.task('task', function() {
  gulp.src(['./*.js', '!./node_modules/**'])
    .pipe(uglify())
    .pipe(gulp.dest('./dist/'));
});
```

Grab all JavaScript files that aren't in the node_modules folder, uglify them, and write them.
This is fastest because nothing in node_modules ever leaves `gulp.src()`


2: Remove things from the stream

**Remove from here on**

![][exclude]

```javascript
var gulpIgnore = require('gulp-ignore');
var uglify = require('gulp-uglify');
var jshint = require('gulp-jshint');

var condition = './gulpfile.js';

gulp.task('task', function() {
  gulp.src('./*.js')
    .pipe(jshint())
    .pipe(gulpIgnore.exclude(condition))
    .pipe(uglify())
    .pipe(gulp.dest('./dist/'));
});
```

Run JSHint on everything, remove gulpfile from the stream, then uglify and write everything else.

3: Filter only matching things

**Include from here on**

![][include]

```javascript
var gulpIgnore = require('gulp-ignore');
var uglify = require('gulp-uglify');
var jshint = require('gulp-jshint');

var condition = './public/**.js';

gulp.task('task', function() {
  gulp.src('./*.js')
    .pipe(jshint())
    .pipe(gulpIgnore.include(condition))
    .pipe(uglify())
    .pipe(gulp.dest('./dist/'));
});
```

Run JSHint on everything, filter to include only files from in the public folder, then uglify and write them.


4: Conditionally filter content, include everything down-stream

**Condition**

![][condition]

```javascript
var gulpif = require('gulp-if');
var uglify = require('gulp-uglify');

var condition = true; // TODO: add business logic

gulp.task('task', function() {
  gulp.src('./src/*.js')
    .pipe(gulpif(condition, uglify()))
    .pipe(gulp.dest('./dist/'));
});
```
Only uglify the content if the condition is true, but send all the files to the dist folder


API
---

### exclude(condition)

Exclude files whose `file.path` matches, include everything else

### include(condition)

Include files whose `file.path` matches, exclude everything else

### condition

Type: `boolean` or [`stat`](http://nodejs.org/api/fs.html#fs_class_fs_stats) object or `function` that takes in a vinyl file and returns a boolean or `RegularExpression` that works on the `file.path`

The condition parameter is any of the conditions supported by [gulp-match](https://github.com/robrich/gulp-match).  The `file.path` is passed into `gulp-match`.

If a function is given, then the function is passed a vinyl `file`. The function should return a `boolean`.


LICENSE
-------

(MIT License)

Copyright (c) 2014 [Richardson & Sons, LLC](http://richardsonandsons.com/)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

[condition]: https://rawgithub.com/robrich/gulp-ignore/master/img/condition.svg
[ternary]: https://rawgithub.com/robrich/gulp-ignore/master/img/ternary.svg
[exclude]: https://rawgithub.com/robrich/gulp-ignore/master/img/exclude.svg
[include]: https://rawgithub.com/robrich/gulp-ignore/master/img/include.svg
[glob]: https://rawgithub.com/robrich/gulp-ignore/master/img/glob.svg
