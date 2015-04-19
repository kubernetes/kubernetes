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

var childProcess = require('child_process');

var promise = require('..').promise;


/**
 * A hash with configuration options for an executed command.
 * <ul>
 * <li>
 * <li>{@code args} - Command line arguments.
 * <li>{@code env} - Command environment; will inherit from the current process
 *     if missing.
 * <li>{@code stdio} - IO configuration for the spawned server process. For
 *     more information, refer to the documentation of
 *     {@code child_process.spawn}.
 * </ul>
 *
 * @typedef {{
 *   args: (!Array.<string>|undefined),
 *   env: (!Object.<string, string>|undefined),
 *   stdio: (string|!Array.<string|number|!Stream|null|undefined>|undefined)
 * }}
 */
var Options;


/**
 * Describes a command's termination conditions.
 * @param {?number} code The exit code, or {@code null} if the command did not
 *     exit normally.
 * @param {?string} signal The signal used to kill the command, or
 *     {@code null}.
 * @constructor
 */
var Result = function(code, signal) {
  /** @type {?number} */
  this.code = code;

  /** @type {?string} */
  this.signal = signal;
};


/** @override */
Result.prototype.toString = function() {
  return 'Result(code=' + this.code + ', signal=' + this.signal + ')';
};



/**
 * Represents a command running in a sub-process.
 * @param {!promise.Promise.<!Result>} result The command result.
 * @constructor
 */
var Command = function(result, onKill) {
  /** @return {boolean} Whether this command is still running. */
  this.isRunning = function() {
    return result.isPending();
  };

  /**
   * @return {!promise.Promise.<!Result>} A promise for the result of this
   *     command.
   */
  this.result = function() {
    return result;
  };

  /**
   * Sends a signal to the underlying process.
   * @param {string=} opt_signal The signal to send; defaults to
   *     {@code SIGTERM}.
   */
  this.kill = function(opt_signal) {
    onKill(opt_signal || 'SIGTERM');
  };
};


// PUBLIC API


/**
 * Spawns a child process. The returned {@link Command} may be used to wait
 * for the process result or to send signals to the process.
 *
 * @param {string} command The executable to spawn.
 * @param {Options=} opt_options The command options.
 * @return {!Command} The launched command.
 */
module.exports = function(command, opt_options) {
  var options = opt_options || {};

  var proc = childProcess.spawn(command, options.args || [], {
    env: options.env || process.env,
    stdio: options.stdio || 'ignore'
  }).once('exit', onExit);

  // This process should not wait on the spawned child, however, we do
  // want to ensure the child is killed when this process exits.
  proc.unref();
  process.once('exit', killCommand);

  var result = promise.defer();
  var cmd = new Command(result.promise, function(signal) {
    if (!result.isPending() || !proc) {
      return;  // No longer running.
    }
    proc.kill(signal);
  });
  return cmd;

  function onExit(code, signal) {
    proc = null;
    process.removeListener('exit', killCommand);
    result.fulfill(new Result(code, signal));
  }

  function killCommand() {
    process.removeListener('exit', killCommand);
    proc && proc.kill('SIGTERM');
  }
};
