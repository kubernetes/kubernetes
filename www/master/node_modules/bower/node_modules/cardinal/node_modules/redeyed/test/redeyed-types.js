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
    var optsi = inspect(opts);
    var result = redeyed(code, opts).code

    this.equals(  result
                , expected
                , util.format('%s: %s => %s', optsi, inspect(code), inspect(expected))
               )
    return this;
  }

  t.end() 
})

test('types', function (t) {
  t.test('\n# Boolean', function (t) {
    t.assertSurrounds('return true;', { 'Boolean': { _default: '$:%' } }, 'return $true%;')
    t.assertSurrounds(  'return true; return false;'
                      , { 'Boolean': { 'false': '#:', _default: '$:%' } }
                      , 'return $true%; return #false%;')
    t.end()
  })

  t.test('\n# Identifier', function (t) {
    t.assertSurrounds('var a = 1;', { 'Identifier': { _default: '$:%' } }, 'var $a% = 1;')
    t.assertSurrounds(  'var a = 1; const b = 2;'
                      , { 'Identifier': { 'b': '#:', _default: '$:%' } }
                      , 'var $a% = 1; const #b% = 2;')
    t.end()
  })

  t.test('\n# Null', function (t) {
    t.assertSurrounds('return null;', { 'Null': { _default: '$:%' } }, 'return $null%;').end()
  })

  t.test('\n# Numeric', function (t) {
    t.assertSurrounds('return 1;', { 'Numeric': { _default: '$:%' } }, 'return $1%;')
    t.assertSurrounds(  'return 1; return 2;'
                      , { 'Numeric': { '2': '#:', _default: '$:%' } }
                      , 'return $1%; return #2%;')
    t.end()
  })

  t.test('\n# Punctuator', function (t) {
    t.assertSurrounds('return 2 * 2;', { 'Punctuator': { _default: '$:%' } }, 'return 2 $*% 2$;%')
    t.assertSurrounds(  'return 2 * 2;'
                      , { 'Punctuator': {'*': '#:', _default: '$:%' } }
                      , 'return 2 #*% 2$;%')
    t.end()
  })

  t.test('\n# String', function (t) {
    t.assertSurrounds('return "hello";', { 'String': { _default: '$:%' } }, 'return $"hello"%;')
    t.assertSurrounds(  'return "hello"; return "world";'
                      , { 'String': { '"world"': '#:', _default: '$:%' } }
                      , 'return $"hello"%; return #"world"%;')
    t.end()
  })
})
