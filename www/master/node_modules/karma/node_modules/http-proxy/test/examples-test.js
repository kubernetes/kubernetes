/*
 * examples.js: Tests which ensure all examples do not throw errors.
 *
 * (C) 2010, Charlie Robbins
 *
 */

var vows = require('vows')
    macros = require('./macros'),
    examples = macros.examples;

//
// Suppress `EADDRINUSE` errors since
// we are just checking for require-time errors
//
process.on('uncaughtException', function (err) {
  if (err.code !== 'EADDRINUSE') {
    throw err;
  }
});

vows.describe('node-http-proxy/examples').addBatch(
  examples.shouldHaveDeps()
).addBatch(
  examples.shouldHaveNoErrors()
).export(module);