var tap = require('tap');
var test = tap.test;

test('no module system', function(t) {
  var fs = require('fs');
  var vm = require('vm');
  var head = fs.readFileSync(require.resolve('../head.js.txt'), 'utf8');
  var src = fs.readFileSync(require.resolve('../'), 'utf8');
  var foot = fs.readFileSync(require.resolve('../foot.js.txt'), 'utf8');
  vm.runInThisContext(head + src + foot, 'semver.js');

  // just some basic poking to see if it did some stuff
  t.type(global.semver, 'object');
  t.type(global.semver.SemVer, 'function');
  t.type(global.semver.Range, 'function');
  t.ok(global.semver.satisfies('1.2.3', '1.2'));
  t.end();
});

