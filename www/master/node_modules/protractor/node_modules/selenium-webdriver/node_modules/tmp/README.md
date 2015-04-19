# Tmp

A simple temporary file and directory creator for [node.js.][1]

[![Build Status](https://secure.travis-ci.org/raszi/node-tmp.png?branch=master)](http://travis-ci.org/raszi/node-tmp)

## About

The main difference between bruce's [node-temp][2] is that mine more
aggressively checks for the existence of the newly created temporary file and
creates the new file with `O_EXCL` instead of simple `O_CREAT | O_RDRW`, so it
is safer.

The API is slightly different as well, Tmp does not yet provide synchronous
calls and all the parameters are optional.

You can set whether you want to remove the temporary file on process exit or
not, and the destination directory can also be set.

## How to install

```bash
npm install tmp
```

## Usage

### File creation

Simple temporary file creation, the file will be unlinked on process exit.

```javascript
var tmp = require('tmp');

tmp.file(function _tempFileCreated(err, path, fd) {
  if (err) throw err;

  console.log("File: ", path);
  console.log("Filedescriptor: ", fd);
});
```

### Directory creation

Simple temporary directory creation, it will be removed on process exit.

If the directory still contains items on process exit, then it won't be removed.

```javascript
var tmp = require('tmp');

tmp.dir(function _tempDirCreated(err, path) {
  if (err) throw err;

  console.log("Dir: ", path);
});
```

If you want to cleanup the directory even when there are entries in it, then
you can pass the `unsafeCleanup` option when creating it.

### Filename generation

It is possible with this library to generate a unique filename in the specified
directory.

```javascript
var tmp = require('tmp');

tmp.tmpName(function _tempNameGenerated(err, path) {
    if (err) throw err;

    console.log("Created temporary filename: ", path);
});
```

## Advanced usage

### File creation

Creates a file with mode `0644`, prefix will be `prefix-` and postfix will be `.txt`.

```javascript
var tmp = require('tmp');

tmp.file({ mode: 0644, prefix: 'prefix-', postfix: '.txt' }, function _tempFileCreated(err, path, fd) {
  if (err) throw err;

  console.log("File: ", path);
  console.log("Filedescriptor: ", fd);
});
```

### Directory creation

Creates a directory with mode `0755`, prefix will be `myTmpDir_`.

```javascript
var tmp = require('tmp');

tmp.dir({ mode: 0750, prefix: 'myTmpDir_' }, function _tempDirCreated(err, path) {
  if (err) throw err;

  console.log("Dir: ", path);
});
```

### mkstemps like

Creates a new temporary directory with mode `0700` and filename like `/tmp/tmp-nk2J1u`.

```javascript
var tmp = require('tmp');

tmp.dir({ template: '/tmp/tmp-XXXXXX' }, function _tempDirCreated(err, path) {
  if (err) throw err;

  console.log("Dir: ", path);
});
```

### Filename generation

The `tmpName()` function accepts the `prefix`, `postfix`, `dir`, etc. parameters also:

```javascript
var tmp = require('tmp');

tmp.tmpName({ template: '/tmp/tmp-XXXXXX' }, function _tempNameGenerated(err, path) {
    if (err) throw err;

    console.log("Created temporary filename: ", path);
});
```

## Graceful cleanup

One may want to cleanup the temporary files even when an uncaught exception
occurs. To enforce this, you can call the `setGracefulCleanup()` method:

```javascript
var tmp = require('tmp');

tmp.setGracefulCleanup();
```

## Options

All options are optional :)

  * `mode`: the file mode to create with, it fallbacks to `0600` on file creation and `0700` on directory creation
  * `prefix`: the optional prefix, fallbacks to `tmp-` if not provided
  * `postfix`: the optional postfix, fallbacks to `.tmp` on file creation
  * `template`: [`mkstemps`][3] like filename template, no default
  * `dir`: the optional temporary directory, fallbacks to system default (guesses from environment)
  * `tries`: how many times should the function try to get a unique filename before giving up, default `3`
  * `keep`: signals that the temporary file or directory should not be deleted on exit, default is `false`, means delete
  * `unsafeCleanup`: recursively removes the created temporary directory, even when it's not empty. default is `false`

[1]: http://nodejs.org/
[2]: https://github.com/bruce/node-temp
[3]: http://www.kernel.org/doc/man-pages/online/pages/man3/mkstemp.3.html
