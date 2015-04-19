# [gulp](https://github.com/wearefractal/gulp)-angular-htmlify

> Change your ng-attributes to data-ng-attributes for HTML5 validation

[![NPM Version](http://img.shields.io/npm/v/gulp-angular-htmlify.svg?style=flat)](https://npmjs.org/package/gulp-angular-htmlify)
[![NPM Downloads](http://img.shields.io/npm/dm/gulp-angular-htmlify.svg?style=flat)](https://npmjs.org/package/gulp-angular-htmlify)
[![Build Status](http://img.shields.io/travis/pgilad/gulp-angular-htmlify.svg?style=flat)](https://travis-ci.org/pgilad/gulp-angular-htmlify)

Ever tried to run an Angular HTML page into w3c validator? Yeah it's a mess.

The solution everyone recommends is to add `data` to all your `ng` directives.
Now with `gulp` this can be easily made part of your build flow, similar to how
ng-min is to creating minifiable Angular syntax.

**Turn this:**
```html
<html ng-app="myApp">
...
<body ng-controller="MainCtrl">
</body>
</html>
```

**Into this:**
```html
<html data-ng-app="myApp">
...
<body data-ng-controller="MainCtrl">
</body>
</html>
```
#### <img src="http://www.w3.org/html/logo/downloads/HTML5_Logo_256.png" alt="HTML5 Valid" width="64" height="64"/>

**gulp-angular-htmlify** looks for `ng-` directives by default and can handle the following cases:
```html
<!-- attribute -->
<ANY ng-directive>
<!-- regular element -->
<ng-directive></ng-directive>
<!-- self closing element -->
<ng-directive />
<!-- custom directive prefix -->
<ui-router></ui-router>
<!-- your name prefix -->
<gilad-cool-loader></gilad-cool-loader>
```

You can add additional prefixes using the option `customPrefixes`.

This plugin plays nice with `type="text/ng-template"` and won't break it.

*Issues with the output should be reported on the [angular-html5 issue tracker](https://github.com/pgilad/angular-html5/issues).*

## Install

Install with [npm](https://npmjs.org/package/gulp-angular-htmlify)

```
npm install --save-dev gulp-angular-htmlify
```

## Example

```js
var gulp = require('gulp');
var htmlify = require('gulp-angular-htmlify');

//simple usage
gulp.task('htmlify', function() {
    gulp.src('public/**/*.html')
        .pipe(htmlify())
        .pipe(gulp.dest('build/'));
});

//using jade as a pre-processer
gulp.task('htmlify', function() {
    gulp.src('partials/**/*.jade')
        .pipe(jade())
        .pipe(htmlify())
        .pipe(gulp.dest('build/'));
});

//Also transforming ui-attributes to data-ui-attributes
gulp.task('htmlify', function() {
    gulp.src('public/**/*.html')
        .pipe(htmlify({
            customPrefixes: ['ui-']
        }))
        .pipe(gulp.dest('build/'));
});
```

## API

### htmlify(params)

`params` is an object that contains the following settings:

#### customPrefixes

Type: `Array`

Default: `[]`

An array to optionally add custom prefixes to the list of converted directives.

For example: `['ui-', 'gijo-']`

By default only `ng-` prefixes are are handled. Any items you add here will be handled as well.

*Note: for this to work - you will need to make sure your directives can load with a `data-` prefix.*

#### verbose

Type: `Boolean`

Default: `false`

Whether to log files that had ng-directives detected and replaced. (Useful for debugging).

Example usage:
```js
//...
.pipe(htmlify({
    verbose:true,
    customPrefixes: ['ui-']
    }))
// --> [gulp] Found and replaced ng-directives in index.html
//...
```

## License

MIT ©2014 **Gilad Peleg**
