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

var AdmZip = require('adm-zip'),
    fs = require('fs'),
    path = require('path'),
    vm = require('vm');

var promise = require('..').promise,
    _base = require('../_base'),
    io = require('../io'),
    extension = require('./extension');


/** @const */
var WEBDRIVER_PREFERENCES_PATH = _base.isDevMode()
    ? path.join(__dirname, '../../../firefox-driver/webdriver.json')
    : path.join(__dirname, '../lib/firefox/webdriver.json');

/** @const */
var WEBDRIVER_EXTENSION_PATH = _base.isDevMode()
    ? path.join(__dirname,
        '../../../../build/javascript/firefox-driver/webdriver.xpi')
    : path.join(__dirname, '../lib/firefox/webdriver.xpi');

/** @const */
var WEBDRIVER_EXTENSION_NAME = 'fxdriver@googlecode.com';



/** @type {Object} */
var defaultPreferences = null;

/**
 * Synchronously loads the default preferences used for the FirefoxDriver.
 * @return {!Object} The default preferences JSON object.
 */
function getDefaultPreferences() {
  if (!defaultPreferences) {
    var contents = fs.readFileSync(WEBDRIVER_PREFERENCES_PATH, 'utf8');
    defaultPreferences = JSON.parse(contents);
  }
  return defaultPreferences;
}


/**
 * Parses a user.js file in a Firefox profile directory.
 * @param {string} f Path to the file to parse.
 * @return {!promise.Promise.<!Object>} A promise for the parsed preferences as
 *     a JSON object. If the file does not exist, an empty object will be
 *     returned.
 */
function loadUserPrefs(f) {
  var done = promise.defer();
  fs.readFile(f, function(err, contents) {
    if (err && err.code === 'ENOENT') {
      done.fulfill({});
      return;
    }

    if (err) {
      done.reject(err);
      return;
    }

    var prefs = {};
    var context = vm.createContext({
      'user_pref': function(key, value) {
        prefs[key] = value;
      }
    });

    vm.runInContext(contents, context, f);
    done.fulfill(prefs);
  });
  return done.promise;
}


/**
 * Copies the properties of one object into another.
 * @param {!Object} a The destination object.
 * @param {!Object} b The source object to apply as a mixin.
 */
function mixin(a, b) {
  Object.keys(b).forEach(function(key) {
    a[key] = b[key];
  });
}


/**
 * @param {!Object} defaults The default preferences to write. Will be
 *     overridden by user.js preferences in the template directory and the
 *     frozen preferences required by WebDriver.
 * @param {string} dir Path to the directory write the file to.
 * @return {!promise.Promise.<string>} A promise for the profile directory,
 *     to be fulfilled when user preferences have been written.
 */
function writeUserPrefs(prefs, dir) {
  var userPrefs = path.join(dir, 'user.js');
  return loadUserPrefs(userPrefs).then(function(overrides) {
    mixin(prefs, overrides);
    mixin(prefs, getDefaultPreferences()['frozen']);

    var contents = Object.keys(prefs).map(function(key) {
      return 'user_pref(' + JSON.stringify(key) + ', ' +
          JSON.stringify(prefs[key]) + ');';
    }).join('\n');

    var done = promise.defer();
    fs.writeFile(userPrefs, contents, function(err) {
      err && done.reject(err) || done.fulfill(dir);
    });
    return done.promise;
  });
};


/**
 * Installs a group of extensions in the given profile directory. If the
 * WebDriver extension is not included in this set, the default version
 * bundled with this package will be installed.
 * @param {!Array.<string>} extensions The extensions to install, as a
 *     path to an unpacked extension directory or a path to a xpi file.
 * @param {string} dir The profile directory to install to.
 * @param {boolean=} opt_excludeWebDriverExt Whether to skip installation of
 *     the default WebDriver extension.
 * @return {!promise.Promise.<string>} A promise for the main profile directory
 *     once all extensions have been installed.
 */
function installExtensions(extensions, dir, opt_excludeWebDriverExt) {
  var hasWebDriver = !!opt_excludeWebDriverExt;
  var next = 0;
  var extensionDir = path.join(dir, 'extensions');
  var done = promise.defer();

  return io.exists(extensionDir).then(function(exists) {
    if (!exists) {
      return promise.checkedNodeCall(fs.mkdir, extensionDir);
    }
  }).then(function() {
    installNext();
    return done.promise;
  });

  function installNext() {
    if (!done.isPending()) {
      return;
    }

    if (next >= extensions.length) {
      if (hasWebDriver) {
        done.fulfill(dir);
      } else {
        install(WEBDRIVER_EXTENSION_PATH);
      }
    } else {
      install(extensions[next++]);
    }
  }

  function install(ext) {
    extension.install(ext, extensionDir).then(function(id) {
      hasWebDriver = hasWebDriver || (id === WEBDRIVER_EXTENSION_NAME);
      installNext();
    }, done.reject);
 }
}


/**
 * Decodes a base64 encoded profile.
 * @param {string} data The base64 encoded string.
 * @return {!promise.Promise.<string>} A promise for the path to the decoded
 *     profile directory.
 */
function decode(data) {
  return io.tmpFile().then(function(file) {
    var buf = new Buffer(data, 'base64');
    return promise.checkedNodeCall(fs.writeFile, file, buf).then(function() {
      return io.tmpDir();
    }).then(function(dir) {
      var zip = new AdmZip(file);
      zip.extractAllTo(dir);  // Sync only? Why?? :-(
      return dir;
    });
  });
}



/**
 * Models a Firefox proifle directory for use with the FirefoxDriver. The
 * {@code Proifle} directory uses an in-memory model until {@link #writeToDisk}
 * is called.
 * @param {string=} opt_dir Path to an existing Firefox profile directory to
 *     use a template for this profile. If not specified, a blank profile will
 *     be used.
 * @constructor
 */
var Profile = function(opt_dir) {
  /** @private {!Object} */
  this.preferences_ = {};

  mixin(this.preferences_, getDefaultPreferences()['mutable']);
  mixin(this.preferences_, getDefaultPreferences()['frozen']);

  /** @private {boolean} */
  this.nativeEventsEnabled_ = true;

  /** @private {(string|undefined)} */
  this.template_ = opt_dir;

  /** @private {number} */
  this.port_ = 0;

  /** @private {!Array.<string>} */
  this.extensions_ = [];
};


/**
 * Registers an extension to be included with this profile.
 * @param {string} extension Path to the extension to include, as either an
 *     unpacked extension directory or the path to a xpi file.
 */
Profile.prototype.addExtension = function(extension) {
  this.extensions_.push(extension);
};


/**
 * Sets a desired preference for this profile.
 * @param {string} key The preference key.
 * @param {(string|number|boolean)} value The preference value.
 * @throws {Error} If attempting to set a frozen preference.
 */
Profile.prototype.setPreference = function(key, value) {
  var frozen = getDefaultPreferences()['frozen'];
  if (frozen.hasOwnProperty(key) && frozen[key] !== value) {
    throw Error('You may not set ' + key + '=' + JSON.stringify(value)
        + '; value is frozen for proper WebDriver functionality ('
        + key + '=' + JSON.stringify(frozen[key]) + ')');
  }
  this.preferences_[key] = value;
};


/**
 * Returns the currently configured value of a profile preference. This does
 * not include any defaults defined in the profile's template directory user.js
 * file (if a template were specified on construction).
 * @param {string} key The desired preference.
 * @return {(string|number|boolean|undefined)} The current value of the
 *     requested preference.
 */
Profile.prototype.getPreference = function(key) {
  return this.preferences_[key];
};


/**
 * @return {number} The port this profile is currently configured to use, or
 *     0 if the port will be selected at random when the profile is written
 *     to disk.
 */
Profile.prototype.getPort = function() {
  return this.port_;
};


/**
 * Sets the port to use for the WebDriver extension loaded by this profile.
 * @param {number} port The desired port, or 0 to use any free port.
 */
Profile.prototype.setPort = function(port) {
  this.port_ = port;
};


/**
 * @return {boolean} Whether the FirefoxDriver is configured to automatically
 *     accept untrusted SSL certificates.
 */
Profile.prototype.acceptUntrustedCerts = function() {
  return !!this.preferences_['webdriver_accept_untrusted_certs'];
};


/**
 * Sets whether the FirefoxDriver should automatically accept untrusted SSL
 * certificates.
 * @param {boolean} value .
 */
Profile.prototype.setAcceptUntrustedCerts = function(value) {
  this.preferences_['webdriver_accept_untrusted_certs'] = !!value;
};


/**
 * Sets whether to assume untrusted certificates come from untrusted issuers.
 * @param {boolean} value .
 */
Profile.prototype.setAssumeUntrustedCertIssuer = function(value) {
  this.preferences_['webdriver_assume_untrusted_issuer'] = !!value;
};


/**
 * @return {boolean} Whether to assume untrusted certs come from untrusted
 *     issuers.
 */
Profile.prototype.assumeUntrustedCertIssuer = function() {
  return !!this.preferences_['webdriver_assume_untrusted_issuer'];
};


/**
 * Sets whether to use native events with this profile.
 * @param {boolean} enabled .
 */
Profile.prototype.setNativeEventsEnabled = function(enabled) {
  this.nativeEventsEnabled_ = enabled;
};


/**
 * Returns whether native events are enabled in this profile.
 * @return {boolean} .
 */
Profile.prototype.nativeEventsEnabled = function() {
  return this.nativeEventsEnabled_;
};


/**
 * Writes this profile to disk.
 * @param {boolean=} opt_excludeWebDriverExt Whether to exclude the WebDriver
 *     extension from the generated profile. Used to reduce the size of an
 *     {@link #encode() encoded profile} since the server will always install
 *     the extension itself.
 * @return {!promise.Promise.<string>} A promise for the path to the new
 *     profile directory.
 */
Profile.prototype.writeToDisk = function(opt_excludeWebDriverExt) {
  var profileDir = io.tmpDir();
  if (this.template_) {
    profileDir = profileDir.then(function(dir) {
      return io.copyDir(
          this.template_, dir, /(parent\.lock|lock|\.parentlock)/);
    }.bind(this));
  }

  // Freeze preferences for async operations.
  var prefs = {};
  mixin(prefs, this.preferences_);

  // Freeze extensions for async operations.
  var extensions = this.extensions_.concat();

  return profileDir.then(function(dir) {
    return writeUserPrefs(prefs, dir);
  }).then(function(dir) {
    return installExtensions(extensions, dir, !!opt_excludeWebDriverExt);
  });
};


/**
 * Encodes this profile as a zipped, base64 encoded directory.
 * @return {!promise.Promise.<string>} A promise for the encoded profile.
 */
Profile.prototype.encode = function() {
  return this.writeToDisk(true).then(function(dir) {
    var zip = new AdmZip();
    zip.addLocalFolder(dir, '');
    return io.tmpFile().then(function(file) {
      zip.writeZip(file);  // Sync! Why oh why :-(
      return promise.checkedNodeCall(fs.readFile, file);
    });
  }).then(function(data) {
    return new Buffer(data).toString('base64');
  });
};


// PUBLIC API


exports.Profile = Profile;
exports.decode = decode;
exports.loadUserPrefs = loadUserPrefs;
