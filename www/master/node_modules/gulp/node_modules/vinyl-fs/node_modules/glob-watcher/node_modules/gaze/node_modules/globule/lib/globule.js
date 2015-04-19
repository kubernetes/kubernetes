/*
 * globule
 * https://github.com/cowboy/node-globule
 *
 * Copyright (c) 2013 "Cowboy" Ben Alman
 * Licensed under the MIT license.
 */

'use strict';

var fs = require('fs');
var path = require('path');

var _ = require('lodash');
var glob = require('glob');
var minimatch = require('minimatch');

// The module.
var globule = exports;

// Process specified wildcard glob patterns or filenames against a
// callback, excluding and uniquing files in the result set.
function processPatterns(patterns, fn) {
  return _.flatten(patterns).reduce(function(result, pattern) {
    if (pattern.indexOf('!') === 0) {
      // If the first character is ! all matches via this pattern should be
      // removed from the result set.
      pattern = pattern.slice(1);
      return _.difference(result, fn(pattern));
    } else {
      // Otherwise, add all matching filepaths to the result set.
      return _.union(result, fn(pattern));
    }
  }, []);
}

// Match a filepath or filepaths against one or more wildcard patterns. Returns
// all matching filepaths. This behaves just like minimatch.match, but supports
// any number of patterns.
globule.match = function(patterns, filepaths, options) {
  // Return empty set if either patterns or filepaths was omitted.
  if (patterns == null || filepaths == null) { return []; }
  // Normalize patterns and filepaths to arrays.
  if (!_.isArray(patterns)) { patterns = [patterns]; }
  if (!_.isArray(filepaths)) { filepaths = [filepaths]; }
  // Return empty set if there are no patterns or filepaths.
  if (patterns.length === 0 || filepaths.length === 0) { return []; }
  // Return all matching filepaths.
  return processPatterns(patterns, function(pattern) {
    return minimatch.match(filepaths, pattern, options || {});
  });
};

// Match a filepath or filepaths against one or more wildcard patterns. Returns
// true if any of the patterns match.
globule.isMatch = function() {
  return globule.match.apply(null, arguments).length > 0;
};

// Return an array of all file paths that match the given wildcard patterns.
globule.find = function() {
  var args = _.toArray(arguments);
  // If the last argument is an options object, remove it from args.
  var options = _.isPlainObject(args[args.length - 1]) ? args.pop() : {};
  // Use the first argument if it's an Array, otherwise use all arguments.
  var patterns = _.isArray(args[0]) ? args[0] : args;
  // Return empty set if there are no patterns or filepaths.
  if (patterns.length === 0) { return []; }
  var srcBase = options.srcBase || options.cwd;
  // Create glob-specific options object.
  var globOptions = _.extend({}, options);
  if (srcBase) {
    globOptions.cwd = srcBase;
  }
  // Get all matching filepaths.
  var matches = processPatterns(patterns, function(pattern) {
    return glob.sync(pattern, globOptions);
  });
  // If srcBase and prefixBase were specified, prefix srcBase to matched paths.
  if (srcBase && options.prefixBase) {
    matches = matches.map(function(filepath) {
      return path.join(srcBase, filepath);
    });
  }
  // Filter result set?
  if (options.filter) {
    matches = matches.filter(function(filepath) {
      // If srcBase was specified but prefixBase was NOT, prefix srcBase
      // temporarily, for filtering.
      if (srcBase && !options.prefixBase) {
        filepath = path.join(srcBase, filepath);
      }
      try {
        if (_.isFunction(options.filter)) {
          return options.filter(filepath, options);
        } else {
          // If the file is of the right type and exists, this should work.
          return fs.statSync(filepath)[options.filter]();
        }
      } catch(err) {
        // Otherwise, it's probably not the right type.
        return false;
      }
    });
  }
  return matches;
};

var pathSeparatorRe = /[\/\\]/g;
var extDotRe = {
  first: /(\.[^\/]*)?$/,
  last: /(\.[^\/\.]*)?$/,
};
function rename(dest, options) {
  // Flatten path?
  if (options.flatten) {
    dest = path.basename(dest);
  }
  // Change the extension?
  if (options.ext) {
    dest = dest.replace(extDotRe[options.extDot], options.ext);
  }
  // Join dest and destBase?
  if (options.destBase) {
    dest = path.join(options.destBase, dest);
  }
  return dest;
}

// Build a mapping of src-dest filepaths from the given set of filepaths.
globule.mapping = function(filepaths, options) {
  // Return empty set if filepaths was omitted.
  if (filepaths == null) { return []; }
  options = _.defaults({}, options, {
    extDot: 'first',
    rename: rename,
  });
  var files = [];
  var fileByDest = {};
  // Find all files matching pattern, using passed-in options.
  filepaths.forEach(function(src) {
    // Generate destination filename.
    var dest = options.rename(src, options);
    // Prepend srcBase to all src paths.
    if (options.srcBase) {
      src = path.join(options.srcBase, src);
    }
    // Normalize filepaths to be unix-style.
    dest = dest.replace(pathSeparatorRe, '/');
    src = src.replace(pathSeparatorRe, '/');
    // Map correct src path to dest path.
    if (fileByDest[dest]) {
      // If dest already exists, push this src onto that dest's src array.
      fileByDest[dest].src.push(src);
    } else {
      // Otherwise create a new src-dest file mapping object.
      files.push({
        src: [src],
        dest: dest,
      });
      // And store a reference for later use.
      fileByDest[dest] = files[files.length - 1];
    }
  });
  return files;
};

// Return a mapping of src-dest filepaths from files matching the given
// wildcard patterns.
globule.findMapping = function(patterns, options) {
  return globule.mapping(globule.find(patterns, options), options);
};
