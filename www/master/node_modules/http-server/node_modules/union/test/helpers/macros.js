/*
 * macros.js: Simple test macros
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var assert = require('assert');

var macros = exports;

macros.assertValidResponse = function (err, res) {
  assert.isTrue(!err);
  assert.equal(res.statusCode, 200);
};

