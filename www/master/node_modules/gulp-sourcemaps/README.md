## gulp-sourcemaps  [![NPM version][npm-image]][npm-url] [![build status][travis-image]][travis-url] [![Test coverage][coveralls-image]][coveralls-url]

### Usage

#### Write inline source maps
Inline source maps are embedded in the source file.

Example:
```javascript
var gulp = require('gulp');
var plugin1 = require('gulp-plugin1');
var plugin2 = require('gulp-plugin2');
var sourcemaps = require('gulp-sourcemaps');

gulp.task('javascript', function() {
  gulp.src('src/**/*.js')
    .pipe(sourcemaps.init())
      .pipe(plugin1())
      .pipe(plugin2())
    .pipe(sourcemaps.write())
    .pipe(gulp.dest('dist'));
});
```

All plugins between `sourcemaps.init()` and `sourcemaps.write()` need to have support for `gulp-sourcemaps`. You can find a list of such plugins in the [wiki](https://github.com/floridoo/gulp-sourcemaps/wiki/Plugins-with-gulp-sourcemaps-support).


#### Write external source map files

To write external source map files, pass a path relative to the destination to `sourcemaps.write()`.

Example:
```javascript
var gulp = require('gulp');
var plugin1 = require('gulp-plugin1');
var plugin2 = require('gulp-plugin2');
var sourcemaps = require('gulp-sourcemaps');

gulp.task('javascript', function() {
  gulp.src('src/**/*.js')
    .pipe(sourcemaps.init())
      .pipe(plugin1())
      .pipe(plugin2())
    .pipe(sourcemaps.write('../maps'))
    .pipe(gulp.dest('dist'));
});
```

#### Load existing source maps

To load existing source maps, pass the option `loadMaps: true` to `sourcemaps.init()`.

Example:
```javascript
var gulp = require('gulp');
var plugin1 = require('gulp-plugin1');
var plugin2 = require('gulp-plugin2');
var sourcemaps = require('gulp-sourcemaps');

gulp.task('javascript', function() {
  gulp.src('src/**/*.js')
    .pipe(sourcemaps.init({loadMaps: true}))
      .pipe(plugin1())
      .pipe(plugin2())
    .pipe(sourcemaps.write())
    .pipe(gulp.dest('dist'));
});
```

#### Handle source files from different directories

Use the `base` option on `gulp.src` to make sure all files are relative to a common base directory.

Example:
```javascript
var gulp = require('gulp');
var plugin1 = require('gulp-plugin1');
var plugin2 = require('gulp-plugin2');
var sourcemaps = require('gulp-sourcemaps');

gulp.task('javascript', function() {
gulp.src(['src/test.js', 'src/testdir/test2.js'], { base: 'src' })
    .pipe(sourcemaps.init())
      .pipe(plugin1())
      .pipe(plugin2())
    .pipe(sourcemaps.write('../maps'))
    .pipe(gulp.dest('dist'));
});
```



### Init Options

- `loadMaps`
  Set to true to load existing maps for source files. Supports the following:
    - inline source maps
    - source map files referenced by a `sourceMappingURL=` comment
    - source map files with the same name (plus .map) in the same directory

- `debug`
  Set this to `true` to output debug messages (e.g. about missing source content).

### Write Options

- `addComment`

  By default a comment containing / referencing the source map is added. Set this to `false` to disable the comment (e.g. if you want to load the source maps by header).

  Example:
  ```javascript
  gulp.task('javascript', function() {
    var stream = gulp.src('src/**/*.js')
      .pipe(sourcemaps.init())
        .pipe(plugin1())
        .pipe(plugin2())
      .pipe(sourcemaps.write('../maps', {addComment: false}))
      .pipe(gulp.dest('dist'));
  });
  ```

- `includeContent`

  By default the source maps include the source code. Pass `false` to use the original files.

  Including the content is the recommended way, because it "just works". When setting this to `false` you have to host the source files and set the correct `sourceRoot`.

- `sourceRoot`

  Set the path where the source files are hosted (use this when `includeContent` is set to `false`). This is an URL (or subpath) relative to the source map, not a local file system path. If you have sources in different subpaths, an absolute path (from the domain root) pointing to the source file root is recommended, or define it with a function.

  Example:
  ```javascript
  gulp.task('javascript', function() {
    var stream = gulp.src('src/**/*.js')
      .pipe(sourcemaps.init())
        .pipe(plugin1())
        .pipe(plugin2())
      .pipe(sourcemaps.write({includeContent: false, sourceRoot: '/src'}))
      .pipe(gulp.dest('dist'));
  });
  ```

  Example (using a function):
  ```javascript
  gulp.task('javascript', function() {
    var stream = gulp.src('src/**/*.js')
      .pipe(sourcemaps.init())
        .pipe(plugin1())
        .pipe(plugin2())
      .pipe(sourcemaps.write({
        includeContent: false,
        sourceRoot: function(file) {
          return '/src';
        }
       }))
      .pipe(gulp.dest('dist'));
  });
  ```

- `sourceMappingURLPrefix`

  Specify a prefix to be prepended onto the source map URL when writing external source maps. Relative paths will have their leading dots stripped.

  Example:
  ```javascript
  gulp.task('javascript', function() {
    var stream = gulp.src('src/**/*.js')
      .pipe(sourcemaps.init())
        .pipe(plugin1())
        .pipe(plugin2())
      .pipe(sourcemaps.write('../maps', {
        sourceMappingURLPrefix: 'https://asset-host.example.com/assets'
      }))
      .pipe(gulp.dest('public/scripts'));
  });
  ```

  Example (using a function):
  ```javascript
  gulp.task('javascript', function() {
    var stream = gulp.src('src/**/*.js')
      .pipe(sourcemaps.init())
        .pipe(plugin1())
        .pipe(plugin2())
      .pipe(sourcemaps.write('../maps', {
        sourceMappingURLPrefix: function(file) {
          return 'https://asset-host.example.com/assets'
        }
      }))
      .pipe(gulp.dest('public/scripts'));
  });
  ```

  This will result in source mapping URL comment like `sourceMappingURL=https://asset-host.example.com/assets/maps/helloworld.js.map`.

- `debug`

  Set this to `true` to output debug messages (e.g. about missing source content).

### Plugin developers only: How to add source map support to plugins

- Generate a source map for the transformation the plugin is applying
- **Important**: Make sure the paths in the generated source map (`file` and `sources`) are relative to `file.base` (e.g. use `file.relative`).
- Apply this source map to the vinyl `file`. E.g. by using [vinyl-sourcemaps-apply](https://github.com/floridoo/vinyl-sourcemaps-apply).
  This combines the source map of this plugin with the source maps coming from plugins further up the chain.
- Add your plugin to the [wiki page](https://github.com/floridoo/gulp-sourcemaps/wiki/Plugins-with-gulp-sourcemaps-support)

#### Example:

```javascript
var through = require('through2');
var applySourceMap = require('vinyl-sourcemaps-apply');
var myTransform = require('myTransform');

module.exports = function(options) {

  function transform(file, encoding, callback) {
    // generate source maps if plugin source-map present
    if (file.sourceMap) {
      options.makeSourceMaps = true;
    }

    // do normal plugin logic
    var result = myTransform(file.contents, options);
    file.contents = new Buffer(result.code);

    // apply source map to the chain
    if (file.sourceMap) {
      applySourceMap(file, result.map);
    }

    this.push(file);
    callback();
  }

  return through.obj(transform);
};
```

[npm-image]: https://img.shields.io/npm/v/gulp-sourcemaps.svg
[npm-url]: https://www.npmjs.com/package/gulp-sourcemaps
[travis-image]: https://img.shields.io/travis/floridoo/gulp-sourcemaps.svg
[travis-url]: https://travis-ci.org/floridoo/gulp-sourcemaps
[coveralls-image]: https://img.shields.io/coveralls/floridoo/gulp-sourcemaps.svg
[coveralls-url]: https://coveralls.io/r/floridoo/gulp-sourcemaps?branch=master
