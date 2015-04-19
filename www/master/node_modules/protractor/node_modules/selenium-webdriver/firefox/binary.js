// Copyright 2014 Selenium committers
// Copyright 2014 Software Freedom Conservancy
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

'use strict';

var child = require('child_process'),
    fs = require('fs'),
    path = require('path'),
    util = require('util');

var promise = require('..').promise,
    _base = require('../_base'),
    io = require('../io'),
    exec = require('../io/exec');



/** @const */
var NO_FOCUS_LIB_X86 = _base.isDevMode() ?
    path.join(__dirname, '../../../../cpp/prebuilt/i386/libnoblur.so') :
    path.join(__dirname, '../lib/firefox/i386/libnoblur.so') ;

/** @const */
var NO_FOCUS_LIB_AMD64 = _base.isDevMode() ?
    path.join(__dirname, '../../../../cpp/prebuilt/amd64/libnoblur64.so') :
    path.join(__dirname, '../lib/firefox/amd64/libnoblur64.so') ;

var X_IGNORE_NO_FOCUS_LIB = 'x_ignore_nofocus.so';

var foundBinary = null;


/**
 * Checks the default Windows Firefox locations in Program Files.
 * @return {!promise.Promise.<?string>} A promise for the located executable.
 *     The promise will resolve to {@code null} if Fireox was not found.
 */
function defaultWindowsLocation() {
  var files = [
    process.env['PROGRAMFILES'] || 'C:\\Program Files',
    process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)'
  ].map(function(prefix) {
    return path.join(prefix, 'Mozilla Firefox\\firefox.exe');
  });
  return io.exists(files[0]).then(function(exists) {
    return exists ? files[0] : io.exists(files[1]).then(function(exists) {
      return exists ? files[1] : null;
    });
  });
}


/**
 * Locates the Firefox binary for the current system.
 * @return {!promise.Promise.<string>} A promise for the located binary. The
 *     promise will be rejected if Firefox cannot be located.
 */
function findFirefox() {
  if (foundBinary) {
    return foundBinary;
  }

  if (process.platform === 'darwin') {
    var osxExe =  '/Applications/Firefox.app/Contents/MacOS/firefox-bin';
    foundBinary = io.exists(osxExe).then(function(exists) {
      return exists ? osxExe : null;
    });
  } else if (process.platform === 'win32') {
    foundBinary = defaultWindowsLocation();
  } else {
    foundBinary = promise.fulfilled(io.findInPath('firefox'));
  }

  return foundBinary = foundBinary.then(function(found) {
    if (found) {
      return found;
    }
    throw Error('Could not locate Firefox on the current system');
  });
}


/**
 * Copies the no focus libs into the given profile directory.
 * @param {string} profileDir Path to the profile directory to install into.
 * @return {!promise.Promise.<string>} The LD_LIBRARY_PATH prefix string to use
 *     for the installed libs.
 */
function installNoFocusLibs(profileDir) {
  var x86 = path.join(profileDir, 'x86');
  var amd64 = path.join(profileDir, 'amd64');

  return mkdir(x86)
      .then(copyLib.bind(null, NO_FOCUS_LIB_X86, x86))
      .then(mkdir.bind(null, amd64))
      .then(copyLib.bind(null, NO_FOCUS_LIB_AMD64, amd64))
      .then(function() {
        return x86 + ':' + amd64;
      });

  function mkdir(dir) {
    return io.exists(dir).then(function(exists) {
      if (!exists) {
        return promise.checkedNodeCall(fs.mkdir, dir);
      }
    });
  }

  function copyLib(src, dir) {
    return io.copy(src, path.join(dir, X_IGNORE_NO_FOCUS_LIB));
  }
}


/**
 * Silently runs Firefox to install a profile directory (which is assumed to be
 * defined in the given environment variables).
 * @param {string} firefox Path to the Firefox executable.
 * @param {!Object.<string, string>} env The environment variables to use.
 * @return {!promise.Promise} A promise for when the profile has been installed.
 */
function installProfile(firefox, env) {
  var installed = promise.defer();
  child.exec(firefox + ' -silent', {env: env, timeout: 180 * 1000},
      function(err) {
        if (err) {
          installed.reject(new Error(
              'Failed to install Firefox profile: ' + err));
          return;
        }
        installed.fulfill();
      });
  return installed.promise;
}


/**
 * Manages a Firefox subprocess configured for use with WebDriver.
 * @param {string=} opt_exe Path to the Firefox binary to use. If not
 *     specified, will attempt to locate Firefox on the current system.
 * @constructor
 */
var Binary = function(opt_exe) {
  /** @private {(string|undefined)} */
  this.exe_ = opt_exe;

  /** @private {!Array.<string>} */
  this.args_ = [];

  /** @private {!Object.<string, string>} */
  this.env_ = {};
  Object.keys(process.env).forEach(function(key) {
    this.env_[key] = process.env[key];
  }.bind(this));
  this.env_['MOZ_CRASHREPORTER_DISABLE'] = '1';
  this.env_['MOZ_NO_REMOTE'] = '1';
  this.env_['NO_EM_RESTART'] = '1';

  /** @private {promise.Promise.<!exec.Command>} */
  this.command_ = null;
};


/**
 * Add arguments to the command line used to start Firefox.
 * @param {...(string|!Array.<string>)} var_args Either the arguments to add as
 *     varargs, or the arguments as an array.
 */
Binary.prototype.addArguments = function(var_args) {
  for (var i = 0; i < arguments.length; i++) {
    if (util.isArray(arguments[i])) {
      this.args_ = this.args_.concat(arguments[i]);
    } else {
      this.args_.push(arguments[i]);
    }
  }
};


/**
 * Launches Firefox and eturns a promise that will be fulfilled when the process
 * terminates.
 * @param {string} profile Path to the profile directory to use.
 * @return {!promise.Promise.<!exec.Result>} A promise for the process result.
 * @throws {Error} If this instance has already been started.
 */
Binary.prototype.launch = function(profile) {
  if (this.command_) {
    throw Error('Firefox is already running');
  }

  var env = {};
  Object.keys(this.env_).forEach(function(key) {
    env[key] = this.env_[key];
  }.bind(this));
  env['XRE_PROFILE_PATH'] = profile;

  var args = ['-foreground'].concat(this.args_);

  var self = this;

  this.command_ = promise.when(this.exe_ || findFirefox(), function(firefox) {
    if (process.platform === 'win32' || process.platform === 'darwin') {
      return firefox;
    }
    return installNoFocusLibs(profile).then(function(ldLibraryPath) {
      env['LD_LIBRARY_PATH'] = ldLibraryPath + ':' + env['LD_LIBRARY_PATH'];
      env['LD_PRELOAD'] = X_IGNORE_NO_FOCUS_LIB;
      return firefox;
    });
  }).then(function(firefox) {
    var install = exec(firefox, {args: ['-silent'], env: env});
    return install.result().then(function(result) {
      if (result.code !== 0) {
        throw Error(
            'Failed to install profile; firefox terminated with ' + result);
      }

      return exec(firefox, {args: args, env: env});
    });
  });

  return this.command_.then(function() {
    // Don't return the actual command handle, just a promise to signal it has
    // been started.
  });
};


/**
 * Kills the managed Firefox process.
 * @return {!promise.Promise} A promise for when the process has terminated.
 */
Binary.prototype.kill = function() {
  if (!this.command_) {
    return promise.defer();  // Not running.
  }
  return this.command_.then(function(command) {
    command.kill();
    return command.result();
  });
};


// PUBLIC API


exports.Binary = Binary;

