var common = require('./common');
var fs = require('fs');
var path = require('path');

// Recursively creates 'dir'
function mkdirSyncRecursive(dir) {
  var baseDir = path.dirname(dir);

  // Base dir exists, no recursion necessary
  if (fs.existsSync(baseDir)) {
    fs.mkdirSync(dir, parseInt('0777', 8));
    return;
  }

  // Base dir does not exist, go recursive
  mkdirSyncRecursive(baseDir);

  // Base dir created, can create dir
  fs.mkdirSync(dir, parseInt('0777', 8));
}

//@
//@ ### mkdir([options ,] dir [, dir ...])
//@ ### mkdir([options ,] dir_array)
//@ Available options:
//@
//@ + `p`: full path (will create intermediate dirs if necessary)
//@
//@ Examples:
//@
//@ ```javascript
//@ mkdir('-p', '/tmp/a/b/c/d', '/tmp/e/f/g');
//@ mkdir('-p', ['/tmp/a/b/c/d', '/tmp/e/f/g']); // same as above
//@ ```
//@
//@ Creates directories.
function _mkdir(options, dirs) {
  options = common.parseOptions(options, {
    'p': 'fullpath'
  });
  if (!dirs)
    common.error('no paths given');

  if (typeof dirs === 'string')
    dirs = [].slice.call(arguments, 1);
  // if it's array leave it as it is

  dirs.forEach(function(dir) {
    if (fs.existsSync(dir)) {
      if (!options.fullpath)
          common.error('path already exists: ' + dir, true);
      return; // skip dir
    }

    // Base dir does not exist, and no -p option given
    var baseDir = path.dirname(dir);
    if (!fs.existsSync(baseDir) && !options.fullpath) {
      common.error('no such file or directory: ' + baseDir, true);
      return; // skip dir
    }

    if (options.fullpath)
      mkdirSyncRecursive(dir);
    else
      fs.mkdirSync(dir, parseInt('0777', 8));
  });
} // mkdir
module.exports = _mkdir;
