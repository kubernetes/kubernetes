/**
 * Usage: node test.js
 */

var mime = require('./mime');
var assert = require('assert');

function eq(a, b) {
  console.log('Test: ' + a + ' === ' + b);
  assert.strictEqual.apply(null, arguments);
}

console.log(Object.keys(mime.extensions).length + ' types');
console.log(Object.keys(mime.types).length + ' extensions\n');

//
// Test mime lookups
//

eq('text/plain', mime.lookup('text.txt'));
eq('text/plain', mime.lookup('.text.txt'));
eq('text/plain', mime.lookup('.txt'));
eq('text/plain', mime.lookup('txt'));
eq('application/octet-stream', mime.lookup('text.nope'));
eq('fallback', mime.lookup('text.fallback', 'fallback'));
eq('application/octet-stream', mime.lookup('constructor'));
eq('text/plain', mime.lookup('TEXT.TXT'));

//
// Test extensions
//

eq('txt', mime.extension(mime.types.text));
eq('html', mime.extension(mime.types.htm));
eq('bin', mime.extension('application/octet-stream'));
eq(undefined, mime.extension('constructor'));

//
// Test node types
//

eq('application/octet-stream', mime.lookup('file.buffer'));
eq('audio/mp4', mime.lookup('file.m4a'));

//
// Test charsets
//

eq('UTF-8', mime.charsets.lookup('text/plain'));
eq(undefined, mime.charsets.lookup(mime.types.js));
eq('fallback', mime.charsets.lookup('application/octet-stream', 'fallback'));

console.log('\nOK');
