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

var os = require('os');


function getLoInterface() {
  var name;
  if (process.platform === 'darwin') {
    name = 'lo0';
  } else if (process.platform === 'linux') {
    name = 'lo';
  }
  return name ? os.networkInterfaces()[name] : null;
}


/**
 * Queries the system network interfaces for an IP address.
 * @param {boolean} loopback Whether to find a loopback address.
 * @param {string=} opt_family The IP family (IPv4 or IPv6). Defaults to IPv4.
 * @return {string} The located IP address or undefined.
 */
function getAddress(loopback, opt_family) {
  var family = opt_family || 'IPv4';
  var addresses = [];

  var interfaces;
  if (loopback) {
    var lo = getLoInterface();
    interfaces = lo ? [lo] : null;
  }
  interfaces = interfaces || os.networkInterfaces();
  for (var key in interfaces) {
    interfaces[key].forEach(function(ipAddress) {
      if (ipAddress.family === family &&
          ipAddress.internal === loopback) {
        addresses.push(ipAddress.address);
      }
    });
  }
  return addresses[0];
}


// PUBLIC API


/**
 * Retrieves the external IP address for this host.
 * @param {string=} opt_family The IP family to retrieve. Defaults to "IPv4".
 * @return {string} The IP address or undefined if not available.
 */
exports.getAddress = function(opt_family) {
  return getAddress(false, opt_family);
};


/**
 * Retrieves a loopback address for this machine.
 * @param {string=} opt_family The IP family to retrieve. Defaults to "IPv4".
 * @return {string} The IP address or undefined if not available.
 */
exports.getLoopbackAddress = function(opt_family) {
  return getAddress(true, opt_family);
};
