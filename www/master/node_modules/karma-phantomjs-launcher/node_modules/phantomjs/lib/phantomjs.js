// Copyright 2013 The Obvious Corporation.

/**
 * @fileoverview Helpers made available via require('phantomjs') once package is
 * installed.
 */

var fs = require('fs')
var path = require('path')
var which = require('which')


/**
 * Where the phantom binary can be found.
 * @type {string}
 */
try {
  exports.path = path.resolve(__dirname, require('./location').location)
} catch(e) {
  // Must be running inside install script.
  exports.path = null
}


/**
 * The version of phantomjs installed by this package.
 * @type {number}
 */
exports.version = '1.9.8'


/**
 * Returns a clean path that helps avoid `which` finding bin files installed
 * by NPM for this repo.
 * @param {string} path
 * @return {string}
 */
exports.cleanPath = function (path) {
  return path
      .replace(/:[^:]*node_modules[^:]*/g, '')
      .replace(/(^|:)\.\/bin(\:|$)/g, ':')
      .replace(/^:+/, '')
      .replace(/:+$/, '')
}


// Make sure the binary is executable.  For some reason doing this inside
// install does not work correctly, likely due to some NPM step.
if (exports.path) {
  try {
    // avoid touching the binary if it's already got the correct permissions
    var st = fs.statSync(exports.path);
    var mode = st.mode | 0555;
    if (mode !== st.mode) {
      fs.chmodSync(exports.path, mode);
    }
  } catch (e) {
    // Just ignore error if we don't have permission.
    // We did our best. Likely because phantomjs was already installed.
  }
}
