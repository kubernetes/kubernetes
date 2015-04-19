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

test('\n undefineds only', function (t) {
  t.assertSurrounds('1 + 2', { Numeric: { _default: undefined } }, '1 + 2')
  t.assertSurrounds('1 + 2', { Numeric: { _default: undefined }, _default: undefined }, '1 + 2')

  t.assertSurrounds(
      'return true'
    , { 'Boolean': { 'true': undefined, 'false': undefined, _default: undefined } , _default: undefined }
    , 'return true'
  )

  t.end()
})

test('\n mixed', function (t) {
  t.assertSurrounds(
      'return true || false'
    , { 'Boolean': { 'true': '&:', 'false': undefined, _default: undefined } , _default: undefined }
    , 'return &true || false'
  )

  t.assertSurrounds(
      'return true || false'
    , { 'Boolean': { 'true': '&:', 'false': undefined, _default: ':?' } , _default: undefined }
    , 'return &true? || false?'
  )

  t.assertSurrounds(
      'return true || false'
    , { 'Boolean': { 'true': '&:', 'false': undefined, _default: undefined } , _default: ':?' }
    , 'return &true? || false'
  )
  
  t.end()
})
