gulp-ng-constant
================

[![Build Status](https://travis-ci.org/guzart/gulp-ng-constant.svg)](https://travis-ci.org/guzart/gulp-ng-constant)

## Information

<table>
<tr>
<td>Package</td><td>gulp-ng-constant</td>
</tr>
<tr>
<td>Description</td>
<td>Plugin for dynamic generation of angular constant modules.<br>
Based of <a href="https://github.com/werk85/grunt-ng-constant">grunt-ng-constant</a></td>
</tr>
<tr>
<td>Node Version</td>
<td>>= 0.10</td>
</tr>
</table>

## Index

1. [Usage](#usage)
  * [Configuration in gulpfile.js](#configuration-in-gulpfilejs)
  * [Configuration in config.json](#configuration-in-configjson)
1. [Options](#options)
  * [name](#optionsname)
  * [~~dest~~](#optionsdest)
  * [stream](#optionsstream)
  * [constants](#optionsconstants)
  * [deps](#optionsdeps)
  * [wrap](#optionswrap)
  * [space](#optionsspace)
  * [template](#optionstemplate)
  * [templatePath](#optionstemplatepath)
1. [Examples](#examples)
  * [Multiple Environments](#multiple-environments)
  * [Stream](#stream)
1. [Special Thanks](#special-thanks)

## Usage

### Configuration in `gulpfile.js`

_**gulpfile.js**_

```javascript
var ngConstant = require('gulp-ng-constant');

gulp.task('config', function () {
  gulp.src('app/config.json')
    .pipe(ngConstant({
      name: 'my.module.config',
      deps: ['ngAnimate'],
      constants: { myPropCnt: 'hola!' },
      wrap: 'amd',
    }))
    // Writes config.js to dist/ folder
    .pipe(gulp.dest('dist'));
});
```

_**app/config.json**_
```json
{
  "myFirstCnt": true,
  "mySecondCnt": { "hello": "world" }
}
```

_**dist/config.js**_ _(output)_

```javascript
define(["require", "exports"], function(require, exports) {
  return angular.module("my.module.config", ["ngAnimate"])
    .constant("myFirstCnt", true)
    .constant("mySecondCnt", { "hello": "world" })
    .constant("myPropCnt", "hola!");
});
```

### Configuration in `config.json`

_**gulpfile.js**_

```javascript
var ngConstant = require('gulp-ng-constant');

gulp.task('config', function () {
  gulp.src('app/config.json')
    .pipe(ngConstant())
    // Writes config.js to dist/ folder
    .pipe(gulp.dest('dist'));
});
```


_**app/config.json**_

```json
{
  "name": "my.module.config",
  "deps": ["ngAnimate"],
  "wrap": "commonjs",
  "constants": {
    "myFirstCnt": true,
    "mySecondCnt": { "hello": "world" }
  }
}
```

_**dist/config.js**_ _(output)_

```javascript
module.exports = angular.module("my.module.config", ["ngAnimate"])
    .constant("myFirstCnt", true)
    .constant("mySecondCnt", { "hello": "world" })
    .constant("myPropCnt", "hola!");
```

## Options

#### options.name

Type: `string`  
Default: `undefined`  
Overrides: `json.name`  

The module name.
This property will override any `name` property defined in the input `json` file.

#### ~~options.dest~~

~~Type: `string`~~  
~~Default: `undefined`~~  
_optional_

~~The path where the generated constant module should be saved.~~  
**DEPRECATED**: To change the vinyl file name use a plugin such as [gulp-rename](https://www.npmjs.com/package/gulp-rename).

#### options.stream

Type: `boolean`  
Default: `false`  
_optional_

If true it will return a gulp stream, which can then be piped other gulp plugins
([Example](#stream)).

#### options.constants

Type: `Object | string`  
Default: `undefined`  
Exends/Overrides: `json.constants`  

Constants to defined in the module.
Can be a `JSON` string or an `Object`.
This property extends the one defined in the input `json` file. If there are
properties with the same name, this properties will override the ones from the
input `json` file.

#### options.deps

Type: `array<string>|boolean`  
Default: `[]`  
Overrides: `json.deps`  
_optional_

An array that specifies the default dependencies a module should have. To add the constants to an existing module, you can set it to `false`.
This property will override any `deps` property defined in the input `json` file.

#### options.wrap

Type: `boolean|string`  
Default: `false`  
Available: `['amd', 'commonjs']`  
_optional_

A boolean to active or deactive the automatic wrapping.
A string who will wrap the result of file, use the
`<%= __ngModule %>` variable to indicate where to put the generated
module content.
A string with 'amd' that wraps the module as an AMD module,
compatible with RequireJS

#### options.space

Type: `string`  
Default: `'\t'`  
_optional_

A string that defines how the JSON.stringify method will prettify your code.

#### options.template

Type: `string`  
Default: _content of [tpls/constant.tpl.ejs](https://github.com/guzart/gulp-ng-constant/blob/master/tpls/constant.tpl.ejs)_  
_optional_

EJS template to apply when creating the output configuration file. The following variables
are passed to the template during render:

  * `moduleName`: the module name (`string`)
  * `deps`: the module dependencies (`array<string>`)
  * `constants`: the module constants (`array<contantObj>`)
    * where a `constantObj` is an object with a `name` and a `value`, both `strings`.

#### options.templatePath

Type: `string`  
Default: `'tpls/constant.tpl.ejs'`  
_optional_

Location of a custom template file for creating the output configuration file. Defaults to the provided constants template file if none provided.

## Examples

### Multiple Environments

_**config.json**_
```json
{
  "development": { "greeting": "Sup!" },
  "production": { "greeting": "Hello" }
}
```

_**gulpfile.js**_
```javascript
var gulp = require('gulp');
var ngConstant = require('gulp-ng-constant');

gulp.task('constants', function () {
  var myConfig = require('./config.json');
  var envConfig = myConfig[process.env];
  return ngConstant({
      constants: envConfig,
      stream: true
    })
    .pipe(gulp.dest('dist'));
});

```

### Stream

```javascript
var gulp = require('gulp');
var ngConstant = require('gulp-ng-constant');
var uglify = require('gulp-uglify');

gulp.task('constants', function () {
  var constants = { hello: 'world' };
  return ngConstant({
      constants: constants,
      stream: true
    })
    .pipe(uglify())  
    .pipe(gulp.dest('dist'));
});
```

## Special Thanks

@alexeygolev, @sabudaye, @ojacquemart, @lukehorvat, @rimian, @andidev
