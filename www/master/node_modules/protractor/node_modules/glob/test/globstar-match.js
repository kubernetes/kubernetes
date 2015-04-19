var Glob = require("../glob.js").Glob
var test = require('tap').test

test('globstar should not have dupe matches', function(t) {
  var pattern = 'a/**/[gh]'
  var g = new Glob(pattern, { cwd: __dirname })
  var matches = []
  g.on('match', function(m) {
    console.error('match %j', m)
    matches.push(m)
  })
  g.on('end', function(set) {
    console.error('set', set)
    matches = matches.sort()
    set = set.sort()
    t.same(matches, set, 'should have same set of matches')
    t.end()
  })
})
