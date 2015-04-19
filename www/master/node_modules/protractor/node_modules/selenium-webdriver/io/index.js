// Copyright 2013 Selenium committers
// Copyright 2013 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

var fs = require('fs'),
    path = require('path'),
    tmp = require('tmp');

var promise = require('..').promise;


var PATH_SEPARATOR = process.platform === 'win32' ? ';' : ':';


// PUBLIC API



/**
 * Copies one file to another.
 * @param {string} src The source file.
 * @param {string} dst The destination file.
 * @return {!promise.Promise.<string>} A promise for the copied file's path.
 */
exports.copy = function(src, dst) {
  var copied = promise.defer();

  var rs = fs.createReadStream(src);
  rs.on('error', copied.reject);
  rs.on('end', function() {
    copied.fulfill(dst);
  });

  var ws = fs.createWriteStream(dst);
  ws.on('error', copied.reject);

  rs.pipe(ws);

  return copied.promise;
};


/**
 * Recursively copies the contents of one directory to another.
 * @param {string} src The source directory to copy.
 * @param {string} dst The directory to copy into.
 * @param {(RegEx|function(string): boolean)=} opt_exclude An exclusion filter
 *     as either a regex or predicate function. All files matching this filter
 *     will not be copied.
 * @return {!promise.Promise.<string>} A promise for the destination
 *     directory's path once all files have been copied.
 */
exports.copyDir = function(src, dst, opt_exclude) {
  var predicate = opt_exclude;
  if (opt_exclude && typeof opt_exclude !== 'function') {
    predicate = function(p) {
      return !opt_exclude.test(p);
    };
  }

  // TODO(jleyba): Make this function completely async.
  if (!fs.existsSync(dst)) {
    fs.mkdirSync(dst);
  }

  var files = fs.readdirSync(src);
  files = files.map(function(file) {
    return path.join(src, file);
  });

  if (predicate) {
    files = files.filter(predicate);
  }

  var results = [];
  files.forEach(function(file) {
    var stats = fs.statSync(file);
    var target = path.join(dst, path.basename(file));

    if (stats.isDirectory()) {
      if (!fs.existsSync(target)) {
        fs.mkdirSync(target, stats.mode);
      }
      results.push(exports.copyDir(file, target, predicate));
    } else {
      results.push(exports.copy(file, target));
    }
  });

  return promise.all(results).then(function() {
    return dst;
  });
};


/**
 * Tests if a file path exists.
 * @param {string} path The path to test.
 * @return {!promise.Promise.<boolean>} A promise for whether the file exists.
 */
exports.exists = function(path) {
  var result = promise.defer();
  fs.exists(path, result.fulfill);
  return result.promise;
};


/**
 * @return {!promise.Promise.<string>} A promise for the path to a temporary
 *     directory.
 * @see https://www.npmjs.org/package/tmp
 */
exports.tmpDir = function() {
  return promise.checkedNodeCall(tmp.dir);
};


/**
 * @return {!promise.Promise.<string>} A promise for the path to a temporary
 *     file.
 * @see https://www.npmjs.org/package/tmp
 */
exports.tmpFile = function() {
  return promise.checkedNodeCall(tmp.file);
};


/**
 * Searches the {@code PATH} environment variable for the given file.
 * @param {string} file The file to locate on the PATH.
 * @param {boolean=} opt_checkCwd Whether to always start with the search with
 *     the current working directory, regardless of whether it is explicitly
 *     listed on the PATH.
 * @return {?string} Path to the located file, or {@code null} if it could
 *     not be found.
 */
exports.findInPath = function(file, opt_checkCwd) {
  if (opt_checkCwd) {
    var tmp = path.join(process.cwd(), file);
    if (fs.existsSync(tmp)) {
      return tmp;
    }
  }

  var dirs = process.env['PATH'].split(PATH_SEPARATOR);
  var found = null;
  dirs.forEach(function(dir) {
    var tmp = path.join(dir, file);
    if (!found && fs.existsSync(tmp)) {
      found = tmp;
    }
  });
  return found;
};
