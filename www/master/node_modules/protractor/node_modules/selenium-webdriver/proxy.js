// Copyright 2013 Selenium committers
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

/**
 * @fileoverview Defines functions for configuring a webdriver proxy:
 * <pre><code>
 * var webdriver = require('selenium-webdriver'),
 *     proxy = require('selenium-webdriver/proxy');
 *
 * var driver = new webdriver.Builder()
 *     .withCapabilities(webdriver.Capabilities.chrome())
 *     .setProxy(proxy.manual({http: 'host:1234'}))
 *     .build();
 * </code></pre>
 */

'use strict';

var util = require('util');



// PUBLIC API


/**
 * Configures WebDriver to bypass all browser proxies.
 * @return {!webdriver.ProxyConfig} A new proxy configuration object.
 */
exports.direct = function() {
  return {proxyType: 'direct'};
};


/**
 * Manually configures the browser proxy.  The following options are
 * supported:
 * <ul>
 * <li>{@code ftp}: Proxy host to use for FTP requests
 * <li>{@code http}: Proxy host to use for HTTP requests
 * <li>{@code https}: Proxy host to use for HTTPS requests
 * <li>{@code bypass}: A list of hosts requests should directly connect to,
 *     bypassing any other proxies for that request. May be specified as a
 *     comma separated string, or a list of strings.
 * </ul>
 *
 * Behavior is undefined for FTP, HTTP, and HTTPS requests if the
 * corresponding key is omitted from the configuration options.
 *
 * @param {{ftp: (string|undefined),
 *          http: (string|undefined),
 *          https: (string|undefined),
 *          bypass: (string|!Array.<string>|undefined)}} options Proxy
 *     configuration options.
 * @return {!webdriver.ProxyConfig} A new proxy configuration object.
 */
exports.manual = function(options) {
  return {
    proxyType: 'manual',
    ftpProxy: options.ftp,
    httpProxy: options.http,
    sslProxy: options.https,
    noProxy: util.isArray(options.bypass) ?
        options.bypass.join(',') : options.bypass
  };
};


/**
 * Configures WebDriver to configure the browser proxy using the PAC file at
 * the given URL.
 * @param {string} url URL for the PAC proxy to use.
 * @return {!webdriver.ProxyConfig} A new proxy configuration object.
 */
exports.pac = function(url) {
  return {
    proxyType: 'pac',
    proxyAutoconfigUrl: url
  };
};


/**
 * Configures WebDriver to use the current system's proxy.
 * @return {!webdriver.ProxyConfig} A new proxy configuration object.
 */
exports.system = function() {
  return {proxyType: 'system'};
};
