'use strict';
/*jshint asi: true*/

var test = require('tap').test
  , util = require('util')
  , redeyed = require('..')

function inspect (obj) {
  return util.inspect(obj, false, 5, true)
}

test('adding custom asserts ... ', function (t) {
  t.constructor.prototype.assertSurrounds = function (code, opts, expected) {
    var result = redeyed(code, opts)
    this.equals(result.code, expected, inspect(code) + ' => ' + inspect(expected))
    return this;
  }

  t.end() 
})

test('\nstring config, Line comments', function (t) {
  var opts = { Line: { _default: '*:&' } };
  t.test('\n# ' + inspect(opts), function (t) {

    t.assertSurrounds(
        '// a comment'
      , opts
      , '*// a comment&'
    )
    t.assertSurrounds(
        '// comment then new line\nif (a == 1) return'
      , opts
      , '*// comment then new line&\nif (a == 1) return'
    )
    t.assertSurrounds(
        'var n = new Test();// some comment after\n//more comment\nvar s = 3;'
      , opts
      , 'var n = new Test();*// some comment after&\n*//more comment&\nvar s = 3;'
    )
    t.end()
  })
})

test('\nstring config, Block comments', function (t) {
  var opts = { Block: { _default: '_:-' } };
  t.test('\n# ' + inspect(opts), function (t) {

    t.assertSurrounds(
        '/* a comment */'
      , opts
      , '_/* a comment */-'
    )
    t.assertSurrounds(
        '/* comment then new line*/\nif (a == 1) /* inline */ return'
      , opts
      , '_/* comment then new line*/-\nif (a == 1) _/* inline */- return'
    )
    t.assertSurrounds(
        'var n = new Test();/* some comment after*/\n/*more comment*/\nvar s = 3;'
      , opts
      , 'var n = new Test();_/* some comment after*/-\n_/*more comment*/-\nvar s = 3;'
    )
    t.assertSurrounds(
        'var a = 4;\n/* Multi line comment\n * Next line\n * and another\n*/ var morecode = "here";'
      , opts
      , 'var a = 4;\n_/* Multi line comment\n * Next line\n * and another\n*/- var morecode = "here";'
    )
    t.end()
  })
})
