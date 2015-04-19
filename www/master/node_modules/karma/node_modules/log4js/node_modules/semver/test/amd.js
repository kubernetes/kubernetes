var tap = require('tap');
var test = tap.test;

test('amd', function(t) {
  global.define = define;
  define.amd = true;
  var defined = null;
  function define(stuff) {
    defined = stuff;
  }
  var fromRequire = require('../');
  t.ok(defined, 'amd function called');
  t.equal(fromRequire, defined, 'amd stuff same as require stuff');
  t.end();
});
