# Chokidar
A neat wrapper around node.js fs.watch / fs.watchFile.

[![NPM](https://nodei.co/npm-dl/chokidar.png)](https://nodei.co/npm/chokidar/)

## Why?
Node.js `fs.watch`:

* Doesn't report filenames on OS X.
* Doesn't report events at all when using editors like Sublime on OS X.
* Doesn't use OS X internals for **fast low-CPU watching on OS X** (no other fs watch module does this!).
* Sometimes reports events twice.
* Has only one non-useful event: `rename`.
* Has [a lot of other issues](https://github.com/joyent/node/search?q=fs.watch&type=Issues)

Node.js `fs.watchFile`:

* Almost as shitty in event tracking.

Other node.js watching libraries:

* Are not using ultra-fast non-polling watcher implementation on OS X

Chokidar resolves these problems.

It is used in
[brunch](http://brunch.io),
[socketstream](http://www.socketstream.org),
and [karma](http://karma-runner.github.io)
and has proven itself in production environments.

## Getting started
Install chokidar via node.js package manager:

    npm install chokidar

Then just require the package in your code:

```javascript
var chokidar = require('chokidar');

var watcher = chokidar.watch('file or dir', {ignored: /[\/\\]\./, persistent: true});

watcher
  .on('add', function(path) {console.log('File', path, 'has been added');})
  .on('addDir', function(path) {console.log('Directory', path, 'has been added');})
  .on('change', function(path) {console.log('File', path, 'has been changed');})
  .on('unlink', function(path) {console.log('File', path, 'has been removed');})
  .on('unlinkDir', function(path) {console.log('Directory', path, 'has been removed');})
  .on('error', function(error) {console.error('Error happened', error);})

// 'add', 'addDir' and 'change' events also receive stat() results as second argument.
// http://nodejs.org/api/fs.html#fs_class_fs_stats
watcher.on('change', function(path, stats) {
  console.log('File', path, 'changed size to', stats.size);
});

watcher.add('new-file');
watcher.add(['new-file-2', 'new-file-3']);

// Only needed if watching is persistent.
watcher.close();

// One-liner
require('chokidar').watch('.', {ignored: /[\/\\]\./}).on('all', function(event, path) {
  console.log(event, path);
});

```

## API
* `chokidar.watch(paths, options)`: takes paths to be watched recursively and options:
    * `options.ignored` (regexp or function) files to be ignored.
      This function or regexp is tested against the **whole path**,
      not just filename. If it is a function with two arguments, it gets called
      twice per path - once with a single argument (the path), second time with
      two arguments (the path and the [`fs.Stats`](http://nodejs.org/api/fs.html#fs_class_fs_stats)
      object of that path).
    * `options.persistent` (default: `false`). Indicates whether the process
    should continue to run as long as files are being watched.
    * `options.ignorePermissionErrors` (default: `false`). Indicates
      whether to watch files that don't have read permissions.
    * `options.ignoreInitial` (default: `false`). Indicates whether chokidar
    should ignore the initial `add` events or not.
    * `options.interval` (default: `100`). Interval of file system polling.
    * `options.binaryInterval` (default: `300`). Interval of file system
    polling for binary files (see extensions in src/is-binary).
    * `options.usePolling` (default: `false` on Linux and Windows, `true` on OS X). Whether to use fs.watchFile
    (backed by polling), or fs.watch. If polling leads to high CPU utilization,
    consider setting this to `false`.

`chokidar.watch()` produces an instance of `FSWatcher`. Methods of `FSWatcher`:

* `.add(file / files)`: Add directories / files for tracking.
Takes an array of strings (file paths) or just one path.
* `.on(event, callback)`: Listen for an FS event.
Available events: `add`, `change`, `unlink`, `error`.
Additionally `all` is available which gets emitted for every `add`, `change` and `unlink`.
* `.close()`: Removes all listeners from watched files.

## License
The MIT license.

Copyright (c) 2013 Paul Miller (http://paulmillr.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
