var common = require('./common');
var fs = require('fs');

// Recursively removes 'dir'
// Adapted from https://github.com/ryanmcgrath/wrench-js
//
// Copyright (c) 2010 Ryan McGrath
// Copyright (c) 2012 Artur Adib
//
// Licensed under the MIT License
// http://www.opensource.org/licenses/mit-license.php
function rmdirSyncRecursive(dir, force) {
  var files;

  files = fs.readdirSync(dir);

  // Loop through and delete everything in the sub-tree after checking it
  for(var i = 0; i < files.length; i++) {
    var file = dir + "/" + files[i],
        currFile = fs.lstatSync(file);

    if(currFile.isDirectory()) { // Recursive function back to the beginning
      rmdirSyncRecursive(file, force);
    }

    else if(currFile.isSymbolicLink()) { // Unlink symlinks
      if (force || isWriteable(file)) {
        try {
          common.unlinkSync(file);
        } catch (e) {
          common.error('could not remove file (code '+e.code+'): ' + file, true);
        }
      }
    }

    else // Assume it's a file - perhaps a try/catch belongs here?
      if (force || isWriteable(file)) {
        try {
          common.unlinkSync(file);
        } catch (e) {
          common.error('could not remove file (code '+e.code+'): ' + file, true);
        }
      }
  }

  // Now that we know everything in the sub-tree has been deleted, we can delete the main directory.
  // Huzzah for the shopkeep.

  var result;
  try {
    result = fs.rmdirSync(dir);
  } catch(e) {
    common.error('could not remove directory (code '+e.code+'): ' + dir, true);
  }

  return result;
} // rmdirSyncRecursive

// Hack to determine if file has write permissions for current user
// Avoids having to check user, group, etc, but it's probably slow
function isWriteable(file) {
  var writePermission = true;
  try {
    var __fd = fs.openSync(file, 'a');
    fs.closeSync(__fd);
  } catch(e) {
    writePermission = false;
  }

  return writePermission;
}

//@
//@ ### rm([options ,] file [, file ...])
//@ ### rm([options ,] file_array)
//@ Available options:
//@
//@ + `-f`: force
//@ + `-r, -R`: recursive
//@
//@ Examples:
//@
//@ ```javascript
//@ rm('-rf', '/tmp/*');
//@ rm('some_file.txt', 'another_file.txt');
//@ rm(['some_file.txt', 'another_file.txt']); // same as above
//@ ```
//@
//@ Removes files. The wildcard `*` is accepted.
function _rm(options, files) {
  options = common.parseOptions(options, {
    'f': 'force',
    'r': 'recursive',
    'R': 'recursive'
  });
  if (!files)
    common.error('no paths given');

  if (typeof files === 'string')
    files = [].slice.call(arguments, 1);
  // if it's array leave it as it is

  files = common.expand(files);

  files.forEach(function(file) {
    if (!fs.existsSync(file)) {
      // Path does not exist, no force flag given
      if (!options.force)
        common.error('no such file or directory: '+file, true);

      return; // skip file
    }

    // If here, path exists

    var stats = fs.lstatSync(file);
    if (stats.isFile() || stats.isSymbolicLink()) {

      // Do not check for file writing permissions
      if (options.force) {
        common.unlinkSync(file);
        return;
      }

      if (isWriteable(file))
        common.unlinkSync(file);
      else
        common.error('permission denied: '+file, true);

      return;
    } // simple file

    // Path is an existing directory, but no -r flag given
    if (stats.isDirectory() && !options.recursive) {
      common.error('path is a directory', true);
      return; // skip path
    }

    // Recursively remove existing directory
    if (stats.isDirectory() && options.recursive) {
      rmdirSyncRecursive(file, options.force);
    }
  }); // forEach(file)
} // rm
module.exports = _rm;
