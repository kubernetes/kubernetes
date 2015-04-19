/**
 * Module dependencies
 */

var crypto = require('crypto');

/**
 * The size ratio between a base64 string and the equivalent byte buffer
 */

var ratio = Math.log(64) / Math.log(256);

/**
 * Make a Base64 string ready for use in URLs
 *
 * @param {String}
 * @returns {String}
 * @api private
 */

function urlReady(str) {
  return str.replace(/\+/g, '_').replace(/\//g, '-');
}

/**
 * Generate an Unique Id
 *
 * @param {Number} length  The number of chars of the uid
 * @param {Number} cb (optional)  Callback for async uid generation
 * @api public
 */

function uid(length, cb) {
  var numbytes = Math.ceil(length * ratio);
  if (typeof cb === 'undefined') {
    return urlReady(crypto.randomBytes(numbytes).toString('base64').slice(0, length));
  } else {
    crypto.randomBytes(numbytes, function(err, bytes) {
       if (err) return cb(err);
       cb(null, urlReady(bytes.toString('base64').slice(0, length)));
    })
  }
}

/**
 * Exports
 */

module.exports = uid;
