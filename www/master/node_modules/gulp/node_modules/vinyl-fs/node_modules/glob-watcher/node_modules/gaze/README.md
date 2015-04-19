# gaze [![Build Status](https://travis-ci.org/shama/gaze.png?branch=master)](https://travis-ci.org/shama/gaze)

A globbing fs.watch wrapper built from the best parts of other fine watch libs.  
Compatible with Node.js 0.10/0.8, Windows, OSX and Linux.

![gaze](http://dontkry.com/images/repos/gaze.png)

## Usage
Install the module with: `npm install gaze` or place into your `package.json`
and run `npm install`.

```javascript
var gaze = require('gaze');

// Watch all .js files/dirs in process.cwd()
gaze('**/*.js', function(err, watcher) {
  // Files have all started watching
  // watcher === this

  // Get all watched files
  console.log(this.watched());

  // On file changed
  this.on('changed', function(filepath) {
    console.log(filepath + ' was changed');
  });

  // On file added
  this.on('added', function(filepath) {
    console.log(filepath + ' was added');
  });

  // On file deleted
  this.on('deleted', function(filepath) {
    console.log(filepath + ' was deleted');
  });

  // On changed/added/deleted
  this.on('all', function(event, filepath) {
    console.log(filepath + ' was ' + event);
  });

  // Get watched files with relative paths
  console.log(this.relative());
});

// Also accepts an array of patterns
gaze(['stylesheets/*.css', 'images/**/*.png'], function() {
  // Add more patterns later to be watched
  this.add(['js/*.js']);
});
```

### Alternate Interface

```javascript
var Gaze = require('gaze').Gaze;

var gaze = new Gaze('**/*');

// Files have all started watching
gaze.on('ready', function(watcher) { });

// A file has been added/changed/deleted has occurred
gaze.on('all', function(event, filepath) { });
```

### Errors

```javascript
gaze('**/*', function() {
  this.on('error', function(err) {
    // Handle error here
  });
});
```

### Minimatch / Glob

See [isaacs's minimatch](https://github.com/isaacs/minimatch) for more
information on glob patterns.

## Documentation

### gaze(patterns, [options], callback)

* `patterns` {String|Array} File patterns to be matched
* `options` {Object}
* `callback` {Function}
  * `err` {Error | null}
  * `watcher` {Object} Instance of the Gaze watcher

### Class: gaze.Gaze

Create a Gaze object by instanting the `gaze.Gaze` class.

```javascript
var Gaze = require('gaze').Gaze;
var gaze = new Gaze(pattern, options, callback);
```

#### Properties

* `options` The options object passed in.
  * `interval` {integer} Interval to pass to fs.watchFile
  * `debounceDelay` {integer} Delay for events called in succession for the same
    file/event

#### Events

* `ready(watcher)` When files have been globbed and watching has begun.
* `all(event, filepath)` When an `added`, `changed` or `deleted` event occurs.
* `added(filepath)` When a file has been added to a watch directory.
* `changed(filepath)` When a file has been changed.
* `deleted(filepath)` When a file has been deleted.
* `renamed(newPath, oldPath)` When a file has been renamed.
* `end()` When the watcher is closed and watches have been removed.
* `error(err)` When an error occurs.
* `nomatch` When no files have been matched.

#### Methods

* `emit(event, [...])` Wrapper for the EventEmitter.emit.
  `added`|`changed`|`deleted` events will also trigger the `all` event.
* `close()` Unwatch all files and reset the watch instance.
* `add(patterns, callback)` Adds file(s) patterns to be watched.
* `remove(filepath)` removes a file or directory from being watched. Does not
  recurse directories.
* `watched()` Returns the currently watched files.
* `relative([dir, unixify])` Returns the currently watched files with relative paths.
  * `dir` {string} Only return relative files for this directory.
  * `unixify` {boolean} Return paths with `/` instead of `\\` if on Windows.

## FAQs

### Why Another `fs.watch` Wrapper?
I liked parts of other `fs.watch` wrappers but none had all the features I
needed. This lib combines the features I needed from other fine watch libs:
Speedy data behavior from
[paulmillr's chokidar](https://github.com/paulmillr/chokidar), API interface
from [mikeal's watch](https://github.com/mikeal/watch) and file globbing using
[isaacs's glob](https://github.com/isaacs/node-glob) which is also used by
[cowboy's Grunt](https://github.com/gruntjs/grunt).

### How do I fix the error `EMFILE: Too many opened files.`?
This is because of your system's max opened file limit. For OSX the default is
very low (256). Increase your limit temporarily with `ulimit -n 10480`, the
number being the new max limit.

## Contributing
In lieu of a formal styleguide, take care to maintain the existing coding style.
Add unit tests for any new or changed functionality. Lint and test your code
using [grunt](http://gruntjs.com/).

## Release History
* 0.5.1 - Use setImmediate (process.nextTick for node v0.8) to defer ready/nomatch events (@amasad).
* 0.5.0 - Process is now kept alive while watching files. Emits a nomatch event when no files are matching.
* 0.4.3 - Track file additions in newly created folders (@brett-shwom).
* 0.4.2 - Fix .remove() method to remove a single file in a directory (@kaelzhang). Fixing Cannot call method 'call' of undefined (@krasimir). Track new file additions within folders (@brett-shwom).
* 0.4.1 - Fix watchDir not respecting close in race condition (@chrisirhc).
* 0.4.0 - Drop support for node v0.6. Use globule for file matching. Avoid node v0.10 path.resolve/join errors. Register new files when added to non-existent folder. Multiple instances can now poll the same files (@jpommerening).
* 0.3.4 - Code clean up. Fix path must be strings errors (@groner). Fix incorrect added events (@groner).
* 0.3.3 - Fix for multiple patterns with negate.
* 0.3.2 - Emit `end` before removeAllListeners.
* 0.3.1 - Fix added events within subfolder patterns.
* 0.3.0 - Handle safewrite events, `forceWatchMethod` option removed, bug fixes and watch optimizations (@rgaskill).
* 0.2.2 - Fix issue where subsequent add calls dont get watched (@samcday). removeAllListeners on close.
* 0.2.1 - Fix issue with invalid `added` events in current working dir.
* 0.2.0 - Support and mark folders with `path.sep`. Add `forceWatchMethod` option. Support `renamed` events.
* 0.1.6 - Recognize the `cwd` option properly
* 0.1.5 - Catch too many open file errors
* 0.1.4 - Really fix the race condition with 2 watches
* 0.1.3 - Fix race condition with 2 watches
* 0.1.2 - Read triggering changed event fix
* 0.1.1 - Minor fixes
* 0.1.0 - Initial release

## License
Copyright (c) 2013 Kyle Robinson Young  
Licensed under the MIT license.
