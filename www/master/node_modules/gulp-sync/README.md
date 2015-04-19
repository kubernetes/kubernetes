# [gulp](http://gulpjs.com)-sync [![Build Status](https://travis-ci.org/kaminaly/gulp-sync.svg?branch=master)](https://travis-ci.org/kaminaly/gulp-sync) [![NPM version](https://badge.fury.io/js/gulp-sync.svg)](http://badge.fury.io/js/gulp-sync)
> sync for dependency tasks of gulp.task method

*this is not gulp plugin*


## Install

```bash
$ npm install --save-dev gulp-sync
```


## Usage

### sync
```js
var gulp = require('gulp');
var gulpsync = require('gulp-sync')(gulp);

gulp.task('default', gulpsync.sync(['a', 'b', 'c']));
```

```js
var gulp = require('gulp');
var gulpsync = require('gulp-sync')(gulp);

gulp.task('default', gulpsync.sync([
    // sync
    'a',
    [
        // async
        'b-1',
        'b-2'
    ],
    [
        // async
        'c-1',
        [
            // sync
            'c-2-1',
            'c-2-2'
        ]
    ]
]));
```

### async
```js
var gulp = require('gulp');
var gulpsync = require('gulp-sync')(gulp);

gulp.task('default', gulpsync.async(['a', 'b', 'c']));
//same gulp.task('default', ['a', 'b', 'c']);
```

```js
var gulp = require('gulp');
var gulpsync = require('gulp-sync')(gulp);

gulp.task('default', gulpsync.async([
    // async
    'a',
    [
        // sync
        'b-1',
        'b-2'
    ],
    [
        // sync
        'c-1',
        [
            // async
            'c-2-1',
            'c-2-2'
        ]
    ]
]));
```

### mix and multiple

~~if you need run multiple, method will required 2nd parameter. that must be unique.~~  
2nd parameter is no longer required.

```js
var gulp = require('gulp');
var gulpsync = require('gulp-sync')(gulp);

gulp.task('debug', gulpsync.async(['a', ['b', 'c']]));
gulp.task('release', gulpsync.sync([['c', 'd'], 'e']));
gulp.task('test', gulpsync.sync(['debug', 'f', 'g']));

gulp.task('default', ['debug']);
```

## API

### sync(tasks, name)

#### tasks

Type: `Array` of `String`

task name list.
required.


#### name

Type: `String`  
Default: `sync group` + `count`

prefix of generated task name  
for instance you can use this parameter for checking the validation



### async(tasks, name)

#### tasks

Type: `Array` of `String`

task name list.
required.


#### name

Type: `String`  
Default: `sync group` + `count`

prefix of generated task name  
for instance you can use this parameter for checking the validation


## License

[MIT](http://opensource.org/licenses/MIT)
