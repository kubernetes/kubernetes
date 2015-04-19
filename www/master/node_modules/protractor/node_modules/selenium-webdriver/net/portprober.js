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

var exec = require('child_process').exec,
    fs = require('fs'),
    net = require('net');

var promise = require('../index').promise;


/**
 * The IANA suggested ephemeral port range.
 * @type {{min: number, max: number}}
 * @const
 * @see http://en.wikipedia.org/wiki/Ephemeral_ports
 */
var DEFAULT_IANA_RANGE = {min: 49152, max: 65535};


/**
 * The epheremal port range for the current system. Lazily computed on first
 * access.
 * @type {webdriver.promise.Promise.<{min: number, max: number}>}
 */
var systemRange = null;


/**
 * Computes the ephemeral port range for the current system. This is based on
 * http://stackoverflow.com/a/924337.
 * @return {webdriver.promise.Promise.<{min: number, max: number}>} A promise
 *     that will resolve to the ephemeral port range of the current system.
 */
function findSystemPortRange() {
  if (systemRange) {
    return systemRange;
  }
  var range = process.platform === 'win32' ?
      findWindowsPortRange() : findUnixPortRange();
  return systemRange = range.thenCatch(function() {
    return DEFAULT_IANA_RANGE;
  });
}


/**
 * Executes a command and returns its output if it succeeds.
 * @param {string} cmd The command to execute.
 * @return {!webdriver.promise.Promise.<string>} A promise that will resolve
 *     with the command's stdout data.
 */
function execute(cmd) {
  var result = promise.defer();
  exec(cmd, function(err, stdout) {
    if (err) {
      result.reject(err);
    } else {
      result.fulfill(stdout);
    }
  });
  return result.promise;
}


/**
 * Computes the ephemeral port range for a Unix-like system.
 * @return {!webdriver.promise.Promise.<{min: number, max: number}>} A promise
 *     that will resolve with the ephemeral port range on the current system.
 */
function findUnixPortRange() {
  var cmd;
  if (process.platform === 'sunos') {
    cmd =
        '/usr/sbin/ndd /dev/tcp tcp_smallest_anon_port tcp_largest_anon_port';
  } else if (fs.existsSync('/proc/sys/net/ipv4/ip_local_port_range')) {
    // Linux
    cmd = 'cat /proc/sys/net/ipv4/ip_local_port_range';
  } else {
    cmd = 'sysctl net.inet.ip.portrange.first net.inet.ip.portrange.last' +
        ' | sed -e "s/.*:\\s*//"';
  }

  return execute(cmd).then(function(stdout) {
    if (!stdout || !stdout.length) return DEFAULT_IANA_RANGE;
    var range = stdout.trim().split(/\s+/).map(Number);
    if (range.some(isNaN)) return DEFAULT_IANA_RANGE;
    return {min: range[0], max: range[1]};
  });
}


/**
 * Computes the ephemeral port range for a Windows system.
 * @return {!webdriver.promise.Promise.<{min: number, max: number}>} A promise
 *     that will resolve with the ephemeral port range on the current system.
 */
function findWindowsPortRange() {
  var deferredRange = promise.defer();
  // First, check if we're running on XP.  If this initial command fails,
  // we just fallback on the default IANA range.
  return execute('cmd.exe /c ver').then(function(stdout) {
    if (/Windows XP/.test(stdout)) {
      // TODO: Try to read these values from the registry.
      return {min: 1025, max: 5000};
    } else {
      return execute('netsh int ipv4 show dynamicport tcp').
          then(function(stdout) {
            /* > netsh int ipv4 show dynamicport tcp
              Protocol tcp Dynamic Port Range
              ---------------------------------
              Start Port : 49152
              Number of Ports : 16384
             */
            var range = stdout.split(/\n/).filter(function(line) {
              return /.*:\s*\d+/.test(line);
            }).map(function(line) {
              return Number(line.split(/:\s*/)[1]);
            });

            return {
              min: range[0],
              max: range[0] + range[1]
            };
          });
    }
  });
}


/**
 * Tests if a port is free.
 * @param {number} port The port to test.
 * @param {string=} opt_host The bound host to test the {@code port} against.
 *     Defaults to {@code INADDR_ANY}.
 * @return {!webdriver.promise.Promise.<boolean>} A promise that will resolve
 *     with whether the port is free.
 */
function isFree(port, opt_host) {
  var result = promise.defer(function() {
    server.cancel();
  });

  var server = net.createServer().on('error', function(e) {
    if (e.code === 'EADDRINUSE') {
      result.fulfill(false);
    } else {
      result.reject(e);
    }
  });

  server.listen(port, opt_host, function() {
    server.close(function() {
      result.fulfill(true);
    });
  });

  return result.promise;
}


/**
 * @param {string=} opt_host The bound host to test the {@code port} against.
 *     Defaults to {@code INADDR_ANY}.
 * @return {!webdriver.promise.Promise.<number>} A promise that will resolve
 *     to a free port. If a port cannot be found, the promise will be
 *     rejected.
 */
function findFreePort(opt_host) {
  return findSystemPortRange().then(function(range) {
    var attempts = 0;
    var deferredPort = promise.defer();
    findPort();
    return deferredPort.promise;

    function findPort() {
      attempts += 1;
      if (attempts > 10) {
        deferredPort.reject(Error('Unable to find a free port'));
      }

      var port = Math.floor(
          Math.random() * (range.max - range.min) + range.min);
      isFree(port, opt_host).then(function(isFree) {
        if (isFree) {
          deferredPort.fulfill(port);
        } else {
          findPort();
        }
      });
    }
  });
}


// PUBLIC API


exports.findFreePort = findFreePort;
exports.isFree = isFree;