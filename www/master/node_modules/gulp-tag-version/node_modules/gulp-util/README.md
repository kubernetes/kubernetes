# gulp-util [![NPM version][npm-image]][npm-url] [![Build Status][travis-image]][travis-url] [![Coveralls Status][coveralls-image]][coveralls-url] [![Dependency Status][depstat-image]][depstat-url]

## Information

<table>
<tr> 
<td>Package</td><td>gulp-util</td>
</tr>
<tr>
<td>Description</td>
<td>Utility functions for gulp plugins</td>
</tr>
<tr>
<td>Node Version</td>
<td>>= 0.9</td>
</tr>
</table>

## Usage

```javascript
var gutil = require('gulp-util');

gutil.log('stuff happened', 'Really it did', gutil.colors.cyan('123'));
gutil.beep();

gutil.replaceExtension('file.coffee', '.js'); // file.js

var opt = {
  name: 'todd',
  file: someGulpFile
};
gutil.template('test <%= name %> <%= file.path %>', opt) // test todd /js/hi.js
```

### log(msg...)

Logs stuff. Already prefixed with [gulp] and all that. Use the right colors for values. If you pass in multiple arguments it will join them by a space.

```
values (files, module names, etc.) = magenta
numbers (times, counts, etc) = cyan
```

### replaceExtension(path, newExtension)

Replaces a file extension in a path. Returns the new path.

### isStream(obj)

Returns true or false if an object is a stream.

### isBuffer(obj)

Returns true or false if an object is a Buffer.

### template(string[, data])

This is a lodash.template function wrapper. You must pass in a valid gulp file object so it is available to the user or it will error. You can not configure any of the delimiters. Look at the [lodash docs](http://lodash.com/docs#template) for more info.

## new File(obj)

This is just [vinyl](https://github.com/wearefractal/vinyl)

```javascript
var file = new gutil.File({
  base: join(__dirname, './fixtures/'),
  cwd: __dirname,
  path: join(__dirname, './fixtures/test.coffee')
});
```

## noop()

Returns a stream that does nothing but pass data straight through.

```javascript
// gulp should be called like this :
// $ gulp --type production
gulp.task('scripts', function() {
  gulp.src('src/**/*.js')
    .pipe(concat('script.js'))
    .pipe(gutil.env.type === 'production' ? uglify() : gutil.noop())
    .pipe(gulp.dest('dist/');
});
```

## buffer(cb)

This is similar to es.wait but instead of buffering text into one string it buffers anything into an array (so very useful for file objects).

Returns a stream that can be piped to.

The stream will emit one data event after the stream piped to it has ended. The data will be the same array passed to the callback.

Callback is optional and receives two arguments: error and data

```javascript
gulp.src('stuff/*.js')
  .pipe(gutil.buffer(function(err, files){
  
  });
```

## new PluginError(pluginName, message[, options])

- pluginName should be the module name of your plugin
- message can be a string or an existing error
- By default the stack will not be shown. Set `options.showStack` to true if you think the stack is important for your error.
- If you pass an error in as the message the stack will be pulled from that, otherwise one will be created.
- Note that if you pass in a custom stack string you need to include the message along with that.

These are all acceptable forms of instantiation:

```javascript
var err = new gutil.PluginError('test', {
  message: 'something broke'
});

var err = new gutil.PluginError({
  plugin: 'test',
  message: 'something broke'
});

var err = new gutil.PluginError('test', 'something broke');

var err = new gutil.PluginError('test', 'something broke', {showStack: true});

var existingError = new Error('OMG');
var err = new gutil.PluginError('test', existingError, {showStack: true});
```

[npm-url]: https://npmjs.org/package/gulp-util
[npm-image]: https://badge.fury.io/js/gulp-util.svg
[travis-url]: https://travis-ci.org/gulpjs/gulp-util
[travis-image]: https://travis-ci.org/gulpjs/gulp-util.svg?branch=master
[coveralls-url]: https://coveralls.io/r/gulpjs/gulp-util
[coveralls-image]: https://coveralls.io/repos/gulpjs/gulp-util/badge.png
[depstat-url]: https://david-dm.org/gulpjs/gulp-util
[depstat-image]: https://david-dm.org/gulpjs/gulp-util.svg
