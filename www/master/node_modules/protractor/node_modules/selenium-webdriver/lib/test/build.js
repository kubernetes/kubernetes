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

'use strict';

var spawn = require('child_process').spawn,
    fs = require('fs'),
    path = require('path');

var promise = require('../..').promise,
    base = require('../../_base');

var projectRoot = path.normalize(path.join(__dirname, '../../../../..'));


function checkIsDevMode() {
  if (!base.isDevMode()) {
    throw Error('Cannot execute build; not running in dev mode');
  }
}


/**
 * Targets that have been previously built.
 * @type {!Object}
 */
var builtTargets = {};


/**
 * @param {!Array.<string>} targets The targets to build.
 * @throws {Error} If not running in dev mode.
 * @constructor
 */
var Build = function(targets) {
  checkIsDevMode();
  this.targets_ = targets;
};


/** @private {boolean} */
Build.prototype.cacheResults_ = false;


/**
 * Configures this build to only execute if it has not previously been
 * run during the life of the current process.
 * @return {!Build} A self reference.
 */
Build.prototype.onlyOnce = function() {
  this.cacheResults_ = true;
  return this;
};


/**
 * Executes the build.
 * @return {!webdriver.promise.Promise} A promise that will be resolved when
 *     the build has completed.
 * @throws {Error} If no targets were specified.
 */
Build.prototype.go = function() {
  var targets = this.targets_;
  if (!targets.length) {
    throw Error('No targets specified');
  }

  // Filter out cached results.
  if (this.cacheResults_) {
    targets = targets.filter(function(target) {
      return !builtTargets.hasOwnProperty(target);
    });

    if (!targets.length) {
      return promise.fulfilled();
    }
  }

  console.log('\nBuilding', targets.join(' '), '...');

  var cmd, args = targets;
  if (process.platform === 'win32') {
    cmd = 'cmd.exe';
    args.unshift('/c', path.join(projectRoot, 'go.bat'));
  } else {
    cmd = path.join(projectRoot, 'go');
  }

  var result = promise.defer();
  spawn(cmd, args, {
    cwd: projectRoot,
    env: process.env,
    stdio: ['ignore', process.stdout, process.stderr]
  }).on('exit', function(code, signal) {
    if (code === 0) {
      targets.forEach(function(target) {
        builtTargets[target] = 1;
      });
      return result.fulfill();
    }

    var msg = 'Unable to build artifacts';
    if (code) {  // May be null.
      msg += '; code=' + code;
    }
    if (signal) {
      msg += '; signal=' + signal;
    }

    result.reject(Error(msg));
  });

  return result.promise;
};


// PUBLIC API


/**
 * Creates a build of the listed targets.
 * @param {...string} var_args The targets to build.
 * @return {!Build} The new build.
 * @throws {Error} If not running in dev mode.
 */
exports.of = function(var_args) {
  var targets = Array.prototype.slice.call(arguments, 0);
  return new Build(targets);
};


/**
 * @return {string} Absolute path of the project's root directory.
 * @throws {Error} If not running in dev mode.
 */
exports.projectRoot = function() {
  checkIsDevMode();
  return projectRoot;
};
