gulp-match ![status](https://secure.travis-ci.org/robrich/gulp-match.png?branch=master)
==========

Does a vinyl file match a condition?  This function checks the condition on the `file.path` of the 
[vinyl-fs](https://github.com/wearefractal/vinyl-fs) file passed to it.

Condition can be a boolean, a function, a regular expression, a glob string (or array of glob strings), or a stat filter object

Used by [gulp-if](https://github.com/robrich/gulp-if) and [gulp-ignore](https://github.com/robrich/gulp-ignore)

## Usage

```javascript
var gulpmatch = require('gulp-match');
var map = require('map-stream');

var condition = true; // TODO: add business logic here

vinylfs.src('path/to/file.js')
  .pipe(map(function (file, cb) {
    var match = gulpmatch(file, condition);
    if (match) {
      // it matched, do stuff
    }
    cb(null, file);
  }));
```

## API

### file

A [vinyl-fs](https://github.com/wearefractal/vinyl-fs) file.

### condition

#### boolean condition

```javascript
var match = gulpmatch(file, true);
```

if the condition parameter is `true` or `false`, results will also be `true` or `false`.

#### function condition

```javascript
var match = gulpmatch(file, function (file) {
  return true;
})
```

if the condition parameter is a function, it will be called, passing in `file` passed to gulp-match.

#### regular expression condition

```javascript
var match = gulpmatch(file, /some\/path\.js/);
```

If the condition is a regular expression, it will be evaluated on the `file.path` passed to gulp-match.

#### glob condition

```javascript
var match = gulpmatch(file, './some/path.js');
```
```javascript
var match = gulpmatch(file, ['./array','!./of','./globs.js']);
```

The globs are passed to [minimatch](https://github.com/isaacs/minimatch).  If the glob matches (or any of the elements in the array match), gulp-match returns `true` else `false`.

#### stat filter condition

```javascript
var match = gulpmatch(file, {isFile:true});
```
```javascript
var match = gulpmatch(file, {isDirectory:false});
```

If the condition is an object with a `isFile` or `isDirectory` property, it'll match the details on the 
[vinyl-fs](https://github.com/wearefractal/vinyl-fs) file's [`stat`](http://nodejs.org/api/fs.html#fs_class_fs_stats) object.

#### else

```javascript
var match = gulpmatch(file, 42);
// match = true
```
```javascript
var match = gulpmatch(file, '');
// match = false
```

If there's no matching rule from the rules above, it'll return `true` for truthy conditions, `false` for falsey conditions (including `undefined`).


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
