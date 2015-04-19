// Copyright 2009 The Closure Library Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS-IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Functions for understanding the version of the browser.
 * This is pulled out of product.js to ensure that only builds that need
 * this functionality actually get it, without having to rely on the compiler
 * to strip out unneeded pieces.
 *
 * TODO(nnaze): Move to more appropriate filename/namespace.
 *
 */


goog.provide('goog.userAgent.product.isVersion');


goog.require('goog.userAgent.product');


/**
 * @return {string} The string that describes the version number of the user
 *     agent product.  This is a string rather than a number because it may
 *     contain 'b', 'a', and so on.
 * @private
 */
goog.userAgent.product.determineVersion_ = function() {
  // All browsers have different ways to detect the version and they all have
  // different naming schemes.

  if (goog.userAgent.product.FIREFOX) {
    // Firefox/2.0.0.1 or Firefox/3.5.3
    return goog.userAgent.product.getFirstRegExpGroup_(/Firefox\/([0-9.]+)/);
  }

  if (goog.userAgent.product.IE || goog.userAgent.product.OPERA) {
    return goog.userAgent.VERSION;
  }

  if (goog.userAgent.product.CHROME) {
    // Chrome/4.0.223.1
    return goog.userAgent.product.getFirstRegExpGroup_(/Chrome\/([0-9.]+)/);
  }

  if (goog.userAgent.product.SAFARI) {
    // Version/5.0.3
    //
    // NOTE: Before version 3, Safari did not report a product version number.
    // The product version number for these browsers will be the empty string.
    // They may be differentiated by WebKit version number in goog.userAgent.
    return goog.userAgent.product.getFirstRegExpGroup_(/Version\/([0-9.]+)/);
  }

  if (goog.userAgent.product.IPHONE || goog.userAgent.product.IPAD) {
    // Mozilla/5.0 (iPod; U; CPU like Mac OS X; en) AppleWebKit/420.1
    // (KHTML, like Gecko) Version/3.0 Mobile/3A100a Safari/419.3
    // Version is the browser version, Mobile is the build number. We combine
    // the version string with the build number: 3.0.3A100a for the example.
    var arr = goog.userAgent.product.execRegExp_(
        /Version\/(\S+).*Mobile\/(\S+)/);
    if (arr) {
      return arr[1] + '.' + arr[2];
    }
  } else if (goog.userAgent.product.ANDROID) {
    // Mozilla/5.0 (Linux; U; Android 0.5; en-us) AppleWebKit/522+
    // (KHTML, like Gecko) Safari/419.3
    //
    // Mozilla/5.0 (Linux; U; Android 1.0; en-us; dream) AppleWebKit/525.10+
    // (KHTML, like Gecko) Version/3.0.4 Mobile Safari/523.12.2
    //
    // Prefer Version number if present, else make do with the OS number
    var version = goog.userAgent.product.getFirstRegExpGroup_(
        /Android\s+([0-9.]+)/);
    if (version) {
      return version;
    }

    return goog.userAgent.product.getFirstRegExpGroup_(/Version\/([0-9.]+)/);
  } else if (goog.userAgent.product.CAMINO) {
    return goog.userAgent.product.getFirstRegExpGroup_(/Camino\/([0-9.]+)/);
  }

  return '';
};


/**
 * Return the first group of the given regex.
 * @param {!RegExp} re Regular expression with at least one group.
 * @return {string} Contents of the first group or an empty string if no match.
 * @private
 */
goog.userAgent.product.getFirstRegExpGroup_ = function(re) {
  var arr = goog.userAgent.product.execRegExp_(re);
  return arr ? arr[1] : '';
};


/**
 * Run regexp's exec() on the userAgent string.
 * @param {!RegExp} re Regular expression.
 * @return {Array} A result array, or null for no match.
 * @private
 */
goog.userAgent.product.execRegExp_ = function(re) {
  return re.exec(goog.userAgent.getUserAgentString());
};


/**
 * The version of the user agent. This is a string because it might contain
 * 'b' (as in beta) as well as multiple dots.
 * @type {string}
 */
goog.userAgent.product.VERSION = goog.userAgent.product.determineVersion_();


/**
 * Whether the user agent product version is higher or the same as the given
 * version.
 *
 * @param {string|number} version The version to check.
 * @return {boolean} Whether the user agent product version is higher or the
 *     same as the given version.
 */
goog.userAgent.product.isVersion = function(version) {
  return goog.string.compareVersions(
      goog.userAgent.product.VERSION, version) >= 0;
};
