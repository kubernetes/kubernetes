var test = require('tape');
var intersect = require('..');

test('intersect', function (t) {
  t.deepEqual(intersect([1], [2]), [], 'empty');
  t.deepEqual(intersect([1, 2], [2]), [2], 'intersection');
  t.end();
});
