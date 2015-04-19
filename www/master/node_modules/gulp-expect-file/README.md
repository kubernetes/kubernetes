# gulp-expect-file [![NPM version][npm-image]][npm-url] [![Build Status][travis-image]][travis-url] [![Dependency Status][daviddm-image]][daviddm-url]
> Expectation on generated files for gulp 3

This plugin is intended for testing other gulp plugin.

![Screen Shot](http://kotas.github.io/gulp-expect-file/screenshot.png)

## Usage

First, install `gulp-expect-file` as a development dependency:

```shell
npm install --save-dev gulp-expect-file
```

Then, add it to your `gulpfile.js`:

```js
var expect = require('gulp-expect-file');

gulp.task('copy', function(){
  gulp.src(['src/foo.txt'])
    .pipe(gulp.dest('dest/'))
    .pipe(expect('dest/foo.txt'))
});
```

## API

### expect(expectation)

#### expectation
Type: `String`, `Array`, `Object` or `Function`

It describes the expectation of files on pipe.

| expectation | meaning |
| ----------- | ------- |
| `"foo.txt"` | Expects `foo.txt` on pipe |
| `"*.txt"`   | Expects any files matching glob `*.txt` on pipe |
| `["a.txt", "b.txt"]` | Expects `a.txt` and `b.txt` both on pipe |
| `{"a.txt": true, "b.txt": true}` | Expects `a.txt` and `b.txt` both on pipe (same as above) |
| `{"foo.txt": "text"}` | Expects `foo.txt` with contents that has `"text"` as substring  |
| `{"foo.txt": /pattern/}` | Expects `foo.txt` with contents that matches `/pattern/` |
| `function (file) { ... }` | Call the tester function for each file on pipe |
| `{"foo.txt": function (file) { ... }}` | Call the tester function for `foo.txt` |

A tester function is called with [vinyl File object](https://github.com/wearefractal/vinyl) of target file.

It can return `true`, `null`, `undefined` for passing that file. `false`, `String` of error message, or any other value will fail testing on that file.

Sync version:
```js
function (file) {
  return /\.txt$/.test(file.path);
}
```

Async version:
```js
function (file, callback) {
  process.nextTick(function () {
    if (/\.txt$/.test(file.path)) {
      callback('not txt file');
    } else {
      callback();
    }
  });
}
```

### expect(options, expectation)

#### options.reportUnexpected
Type: `Boolean`
Default: `true`

If true, files not matching any expectation will be reported as failure.

For example, if `a.txt` and `b.txt` are on the pipe, `expect(['a.txt'])` will report that `b.txt` is unexpected.

```js
gulp.src(['a.txt', 'b.txt'])
  .pipe(expect(['a.txt']))

// => FAIL: b.txt unexpected
```

#### options.reportMissing
Type: `Boolean`
Default: `true`

If true, expected files that are not on the pipe will be reported as failure.

For example, if `a.txt` is on the pipe, `expect(['a.txt', 'b.txt'])` will report that `b.txt` is missing.

```js
gulp.src(['a.txt'])
  .pipe(expect(['a.txt', 'b.txt']))

// => FAIL: Missing 1 expected files: b.txt
```

#### options.checkRealFile
Type: `Boolean`
Default: `false`

If true, it also checks if the real file exists on the file system by `fs.exists()`.

```js
gulp.src(['exist.txt', 'nonexist.txt'])
  .pipe(expect({ checkRealFile: true }, '*.txt'))

// => FAIL: nonexist.txt not exists on filesystem
```

#### options.errorOnFailure
Type: `Boolean`
Default: `false`

If true, it emits `error` event when expectations got failed.

```js
gulp.src(['a.txt'])
  .pipe(expect({ errorOnFailure: true }, ['b.txt']))
    .on('error', function (err) { console.error(err); })
```

#### options.silent
Type: `Boolean`
Default: `false`

If true, it does not report any results.

#### options.verbose
Type: `Boolean`
Default: `false`

If true, it reports files that passed the expectation.

### expect.real([options,] expectation)

This is just a shortcut for `expect({ checkRealFile: true }, expectation)`.


[npm-url]: https://npmjs.org/package/gulp-expect-file
[npm-image]: https://badge.fury.io/js/gulp-expect-file.png
[travis-url]: https://travis-ci.org/kotas/gulp-expect-file
[travis-image]: https://travis-ci.org/kotas/gulp-expect-file.png?branch=master
[daviddm-url]: https://david-dm.org/kotas/gulp-expect-file
[daviddm-image]: https://david-dm.org/kotas/gulp-expect-file.png?theme=shields.io
