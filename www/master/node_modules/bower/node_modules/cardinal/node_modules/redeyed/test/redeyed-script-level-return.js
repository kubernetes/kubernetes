'use strict';
/*jshint asi: true*/

var test = require('tap').test
  , util = require('util')
  , redeyed = require('..')

function inspect (obj) {
  return util.inspect(obj, false, 5, true)
}

test('properly handles script level return -- no blow up', function (t) {
  var code = [
      , 'return 1;'
      ].join('\n')
    , opts = { Keyword: { 'return': '%:^' } }
    , expected = '\n%return^ 1;'
    , res = redeyed(code, opts).code

  t.equals(res, expected, inspect(code) + ' opts: ' + inspect(opts) + ' => ' + inspect(expected))
  t.end()
})
