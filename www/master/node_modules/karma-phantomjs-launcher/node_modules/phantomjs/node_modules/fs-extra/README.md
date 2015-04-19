Node.js: fs-extra
=================

[![build status](https://secure.travis-ci.org/jprichardson/node-fs-extra.svg)](http://travis-ci.org/jprichardson/node-fs-extra)
[![downloads per month](http://img.shields.io/npm/dm/fs-extra.svg)](https://www.npmjs.org/package/fs-extra)
[![Coverage Status](https://img.shields.io/coveralls/jprichardson/node-fs-extra.svg)](https://coveralls.io/r/jprichardson/node-fs-extra)

This module adds a few extra file system methods that aren't included in the native `fs` module. It is a drop in replacement for `fs`.



Why?
----

I got tired of including `mkdirp`, `rimraf`, and `cp -r` in most of my projects. 




Installation
------------

    npm install --save fs-extra



Usage
-----

`fs-extra` is a drop in replacement for native `fs`. All methods in `fs` are unmodified and attached to `fs-extra`.

You don't ever need to include the original `fs` module again:

```javascript
var fs = require('fs') //this is no longer necessary
```

you can now do this:

```javascript
var fs = require('fs-extra') //var fs = require('fs')
```

or if you prefer to make it clear that you're using `fs-extra` and not `fs`, you may want 
to do this:

```javascript
//var fs = require('fs')
var fse = require('fs-extra')
```

you can also keep both, but it's redundant:

```javascript
var fs = require('fs')
var fse = require('fs-extra')
```



Methods
-------

**NOTE:** You can still use the native Node.js methods. They are copied over to `fs-extra`.


### copy(src, dest, [options], callback)

Copy a file or directory. The directory can have contents. Like `cp -r`.

Sync: `copySync()`


Examples:

```javascript
var fs = require('fs-extra')

fs.copy('/tmp/myfile', '/tmp/mynewfile', function(err) {
  if (err) return console.error(err)
  console.log("success!")
}) //copies file

fs.copy('/tmp/mydir', '/tmp/mynewdir', function(err) {
  if (err) return console.error(err)
  console.log("success!")
}) //copies directory, even if it has subdirectories or files
```


### ensureFile(file, callback) 

Ensures that the file exists. If the file that is requested to be created is in directories that do not exist, these directories are created. If the file already exists, it is **NOT MODIFIED**.

Alias: `createFile()`

Sync: `createFileSync()`,`ensureFileSync()`


Example:

```javascript
var fs = require('fs-extra')

var file = '/tmp/this/path/does/not/exist/file.txt'
fs.ensureFile(file, function(err) {
  console.log(err) // => null
  //file has now been created, including the directory it is to be placed in
})
```


### ensureDir(dir, callback) 

Ensures that the directory exists. If the directory structure does not exist, it is created.

Sync: `ensureDirSync()`


Example:

```javascript
var fs = require('fs-extra')

var dir = '/tmp/this/path/does/not/exist'
fs.ensureDir(dir, function(err) {
  console.log(err) // => null
  //dir has now been created, including the directory it is to be placed in
})
```



### mkdirs(dir, callback) 

Creates a directory. If the parent hierarchy doesn't exist, it's created. Like `mkdir -p`.

Alias: `mkdirp()`

Sync: `mkdirsSync()` / `mkdirpSync()`


Examples:

```javascript
var fs = require('fs-extra')

fs.mkdirs('/tmp/some/long/path/that/prob/doesnt/exist', function(err) {
  if (err) return console.error(err)
  console.log("success!")
})

fs.mkdirsSync('/tmp/another/path')
```


### move(src, dest, [options], callback)

Moves a file or directory, even across devices.

Options:  
clobber (boolean): overwrite existing file or directory  
limit (number): number of concurrent moves, see ncp for more information

Example:

```javascript
var fs = require('fs-extra')

fs.move('/tmp/somefile', '/tmp/does/not/exist/yet/somefile', function(err) {
  if (err) return console.error(err)
  console.log("success!")
})
```


### outputFile(file, data, callback)

Almost the same as `writeFile` (i.e. it [overwrites](http://pages.citebite.com/v2o5n8l2f5reb)), except that if the parent directory does not exist, it's created.

Sync: `outputFileSync()`


Example:

```javascript
var fs = require('fs-extra')
var file = '/tmp/this/path/does/not/exist/file.txt'

fs.outputFile(file, 'hello!', function(err) {
  console.log(err) // => null

  fs.readFile(file, 'utf8', function(err, data) {
    console.log(data) // => hello!
  })
})
```



### outputJson(file, data, callback)

Almost the same as `writeJson`, except that if the directory does not exist, it's created.

Alias: `outputJSON()`

Sync: `outputJsonSync()`, `outputJSONSync()`


Example:

```javascript
var fs = require('fs-extra')
var file = '/tmp/this/path/does/not/exist/file.txt'

fs.outputJson(file, {name: 'JP'}, function(err) {
  console.log(err) // => null

  fs.readJson(file, function(err, data) {
    console.log(data.name) // => JP
  })
})
```



### readJson(file, [options], callback) 

Reads a JSON file and then parses it into an object. `options` are the same that you'd pass to `fs.readFile`.

Alias: `readJSON()`

Sync: `readJsonSync()`, `readJSONSync()`


Example:

```javascript
var fs = require('fs-extra')

fs.readJson('./package.json', function(err, packageObj) {
  console.log(packageObj.version) // => 0.1.3
})
```

`readJsonSync()` can take a `throws` option set to `false` and it won't throw if the JSON is invalid. Example:

```js
var fs = require('fs-extra')
var file = path.join('/tmp/some-invalid.json')
var data = "{not valid JSON"
fs.writeFileSync(file, data)

var obj = fs.readJsonSync(file, {throws: false})
console.log(obj) // => null
```


### remove(dir, callback)

Removes a file or directory. The directory can have contents. Like `rm -rf`.

Alias: `delete()`

Sync: `removeSync()` / `deleteSync()`


Examples:

```javascript
var fs = require('fs-extra')

fs.remove('/tmp/myfile', function(err) {
  if (err) return console.error(err)
  
  console.log("success!")
})

fs.removeSync('/home/jprichardson') //I just deleted my entire HOME directory. 
```



### writeJson(file, object, [options], callback) 

Writes an object to a JSON file. `options` are the same that you'd pass to `fs.readFile`.

Alias: `writeJSON()`

Sync: `writeJsonSync()`, `writeJSONSync()`

Example:

```javascript
var fs = require('fs-extra')
fs.writeJson('./package.json', {name: 'fs-extra'}, function(err) {
  console.log(err)
})
```


Third Party
-----------

### Promises

Use [Bluebird](https://github.com/petkaantonov/bluebird). See https://github.com/petkaantonov/bluebird/blob/master/API.md#promisification. `fs-extra` is
explicitly listed as supported.

```js
var Promise = require("bluebird")
var fs = Promise.promisifyAll(require("fs-extra"))
```

Or you can use the package [`fs-extra-promise`](https://github.com/overlookmotel/fs-extra-promise) that marries the two together.


### TypeScript

If you like TypeScript, you can use `fs-extra` with it: https://github.com/borisyankov/DefinitelyTyped/tree/master/fs-extra


### File / Directory Watching

If you want to watch for changes to files or directories, then you should use [chokidar](https://github.com/paulmillr/chokidar).


### Misc.

- [mfs](https://github.com/cadorn/mfs) - Monitor your fs-extra calls.




Naming
------

I put a lot of thought into the naming of these functions. Inspired by @coolaj86's request. So he deserves much of the credit for raising the issue. See discussion(s) here:

* https://github.com/jprichardson/node-fs-extra/issues/2
* https://github.com/flatiron/utile/issues/11
* https://github.com/ryanmcgrath/wrench-js/issues/29
* https://github.com/substack/node-mkdirp/issues/17

First, I believe that in as many cases as possible, the [Node.js naming schemes](http://nodejs.org/api/fs.html) should be chosen. However, there are problems with the Node.js own naming schemes.

For example, `fs.readFile()` and `fs.readdir()`: the **F** is capitalized in *File* and the **d** is not capitalized in *dir*. Perhaps a bit pedantic, but they should still be consistent. Also, Node.js has chosen a lot of POSIX naming schemes, which I believe is great. See: `fs.mkdir()`, `fs.rmdir()`, `fs.chown()`, etc.

We have a dilemma though. How do you consistently name methods that perform the following POSIX commands: `cp`, `cp -r`, `mkdir -p`, and `rm -rf`?

My perspective: when in doubt, err on the side of simplicity. A directory is just a hierarchical grouping of directories and files. Consider that for a moment. So when you want to copy it or remove it, in most cases you'll want to copy or remove all of its contents. When you want to create a directory, if the directory that it's suppose to be contained in does not exist, then in most cases you'll want to create that too. 

So, if you want to remove a file or a directory regardless of whether it has contents, just call `fs.remove(path)` or its alias `fs.delete(path)`. If you want to copy a file or a directory whether it has contents, just call `fs.copy(source, destination)`. If you want to create a directory regardless of whether its parent directories exist, just call `fs.mkdirs(path)` or `fs.mkdirp(path)`. 


Credit
------

`fs-extra` wouldn't be possible without using the modules from the following authors:

- [Isaac Shlueter](https://github.com/isaacs)
- [Charlie McConnel](https://github.com/avianflu)
- [James Halliday](https://github.com/substack)
- [Andrew Kelley](https://github.com/andrewrk)


Contributions
-------------

If you want to contribute, please add a test. Also, don't change the version in `package.json`.


### Contributors

- [*] [JP Richardson](https://github.com/jprichardson)
- [*] [Mike McNeil](https://github.com/mikermcneil)
- [*] [Ian Crowther](https://github.com/iancrowther)
- [*] [Stephen Mathieson](https://github.com/stephenmathieson)
- [*] [Srirangan](https://github.com/Srirangan)
- [*] [Uli Köhler](https://github.com/ulikoehler)
- [2] [Sylvain Cleymans](https://github.com/Ackar)
- [1] [Jim Higson](https://github.com/jimhigson)
- [1] [PatrickJS](https://github.com/gdi2290)
- [1] [Michael Tiller](https://github.com/xogeny)
- [1] [yibuyisheng](https://github.com/yibuyisheng)
- [1] [overlookmotel](https://github.com/overlookmotel)
- `<your name here>`




License
-------


Licensed under MIT

Copyright (c) 2011-2014 JP Richardson

[1]: http://nodejs.org/docs/latest/api/fs.html 


[jsonfile]: https://github.com/jprichardson/node-jsonfile








