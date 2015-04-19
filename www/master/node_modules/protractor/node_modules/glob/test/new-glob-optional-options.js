var Glob = require('../glob.js').Glob;
var test = require('tap').test;

test('new glob, with cb, and no options', function (t) {
  new Glob(__filename, function(er, results) {
    if (er) throw er;
    t.same(results, [__filename]);
    t.end();
  });
});
