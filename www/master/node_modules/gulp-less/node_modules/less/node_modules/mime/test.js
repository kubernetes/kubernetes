/**
 * Usage: node test.js
 */

var mime = require('./mime');
var assert = require('assert');
var path = require('path');

function eq(a, b) {
  console.log('Test: ' + a + ' === ' + b);
  assert.strictEqual.apply(null, arguments);
}

console.log(Object.keys(mime.extensions).length + ' types');
console.log(Object.keys(mime.types).length + ' extensions\n');

//
// Test mime lookups
//

eq('text/plain', mime.lookup('text.txt'));     // normal file
eq('text/plain', mime.lookup('TEXT.TXT'));     // uppercase
eq('text/plain', mime.lookup('dir/text.txt')); // dir + file
eq('text/plain', mime.lookup('.text.txt'));    // hidden file
eq('text/plain', mime.lookup('.txt'));         // nameless
eq('text/plain', mime.lookup('txt'));          // extension-only
eq('text/plain', mime.lookup('/txt'));         // extension-less ()
eq('text/plain', mime.lookup('\\txt'));        // Windows, extension-less
eq('application/octet-stream', mime.lookup('text.nope')); // unrecognized
eq('fallback', mime.lookup('text.fallback', 'fallback')); // alternate default

//
// Test extensions
//

eq('txt', mime.extension(mime.types.text));
eq('html', mime.extension(mime.types.htm));
eq('bin', mime.extension('application/octet-stream'));
eq('bin', mime.extension('application/octet-stream '));
eq('html', mime.extension(' text/html; charset=UTF-8'));
eq('html', mime.extension('text/html; charset=UTF-8 '));
eq('html', mime.extension('text/html; charset=UTF-8'));
eq('html', mime.extension('text/html ; charset=UTF-8'));
eq('html', mime.extension('text/html;charset=UTF-8'));
eq('html', mime.extension('text/Html;charset=UTF-8'));
eq(undefined, mime.extension('unrecognized'));

//
// Test node.types lookups
//

eq('application/font-woff', mime.lookup('file.woff'));
eq('application/octet-stream', mime.lookup('file.buffer'));
eq('audio/mp4', mime.lookup('file.m4a'));
eq('font/opentype', mime.lookup('file.otf'));

//
// Test charsets
//

eq('UTF-8', mime.charsets.lookup('text/plain'));
eq(undefined, mime.charsets.lookup(mime.types.js));
eq('fallback', mime.charsets.lookup('application/octet-stream', 'fallback'));

//
// Test for overlaps between mime.types and node.types
//

var apacheTypes = new mime.Mime(), nodeTypes = new mime.Mime();
apacheTypes.load(path.join(__dirname, 'types/mime.types'));
nodeTypes.load(path.join(__dirname, 'types/node.types'));

var keys = [].concat(Object.keys(apacheTypes.types))
             .concat(Object.keys(nodeTypes.types));
keys.sort();
for (var i = 1; i < keys.length; i++) {
  if (keys[i] == keys[i-1]) {
    console.warn('Warning: ' +
      'node.types defines ' + keys[i] + '->' + nodeTypes.types[keys[i]] +
      ', mime.types defines ' + keys[i] + '->' + apacheTypes.types[keys[i]]);
  }
}

console.log('\nOK');
