gulp-livereload
===

[![Build Status](http://img.shields.io/travis/vohof/gulp-livereload/master.svg?style=flat)](https://travis-ci.org/vohof/gulp-livereload) ![Livereload downloads ](http://img.shields.io/npm/dm/gulp-livereload.svg?style=flat) [![MIT Licensed](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](#license)

A [gulp](https://github.com/gulpjs/gulp) plugin for livereload best used with the [livereload chrome extension](https://chrome.google.com/webstore/detail/livereload/jnihajbhpnppcggbcgedagnkighmdlei).

Install
---

```
npm install --save-dev gulp-livereload
```

### livereload(port/server)
### livereload(options)
### livereload(port/server, options)
### livereload()


Create a `Transform` stream and listen to the port or a `tiny-lr.Server` instance.  If none is passed, a livereload server is automatically started listening on port `35729`.


**options.silent**

Suppress all debug messages. Default is `false`.

**options.auto**

Automatically start a livereload server. Default is `true`.

**options.key**<br>
**options.cert**

Options are also passed to `tinylr`. Including a `key` and `cert` will create an HTTPS server.

### livereload.listen(port/server)
### livereload.listen(options)
### livereload.listen(port/server, options)
### livereload.listen()

Listen to the port or a `tiny-lr.Server` instance.  If none is passed, a livereload server is automatically started listening on port `35729`. Does not create a stream.

### livereload.changed(filepath, port/server)
### livereload.changed(filepath)
### livereload.changed()

Notify a change.

Sample Usages
---

use as a stream:

```javascript
var gulp = require('gulp'),
    less = require('gulp-less'),
    livereload = require('gulp-livereload'),
    watch = require('gulp-watch');

gulp.task('less', function() {
  gulp.src('less/*.less')
    .pipe(watch())
    .pipe(less())
    .pipe(gulp.dest('css'))
    .pipe(livereload());
});
```

use with `gulp.watch`

```javascript
var gulp = require('gulp'),
    less = require('gulp-less'),
    livereload = require('gulp-livereload');

gulp.task('less', function() {
  gulp.src('less/*.less')
    .pipe(less())
    .pipe(gulp.dest('build/css'));
});

gulp.task('watch', function() {
  livereload.listen();
  gulp.watch('build/**').on('change', livereload.changed);
});
```

start lr server at your own will

```javascript
var gulp = require('gulp'),
    less = require('gulp-less'),
    livereload = require('gulp-livereload');

gulp.task('less', function() {
  gulp.src('less/*.less')
    .pipe(less())
    .pipe(gulp.dest('css'))
    .pipe(livereload({ auto: false }));
});

gulp.task('watch', function() {
  livereload.listen();
  gulp.watch('build/**', ['less']);
});
```

### Example usage with static server

```javascript
var livereload = require('gulp-livereload'),
    dest = 'build';

gulp.task('server', function(next) {
  var connect = require('connect'),
      server = connect();
  server.use(connect.static(dest)).listen(process.env.PORT || 80, next);
});

gulp.task('watch', ['server'], function() {
  var server = livereload();
  gulp.watch(dest + '/**').on('change', function(file) {
      server.changed(file.path);
  });
});
```

License
---

The MIT License (MIT)

Copyright (c) 2014 Cyrus David

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
