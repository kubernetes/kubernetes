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
    var result = redeyed(code, opts).code
    this.equals(result, expected, inspect(code) + ' => ' + inspect(expected))
    return this;
  }

  t.end() 
})

test('\nmixed config, keywords', function (t) {
  var opts001 = { 
    Keyword: { 
        'this': function (s) { return '_' + s; }
      , 'if': { _before: '^' }
      , _default: '*:&' 
    } 
  };
  t.test('\n# ' + inspect(opts001), function (t) {
   t.assertSurrounds('if (this.hello) return "world";', opts001, '^if& (_this.hello) *return& "world";').end()
  })
  
  var opts002 = { 
      Keyword: { 
          'this': function (s) { return '_' + s; }
        , 'if': { _before: '^' }
        , 'return': ':)'
        , _default: ':&' 
      } 
    , _default: '*:&' 
  };
  t.test('\n# ' + inspect(opts002), function (t) {
   t.assertSurrounds('if (this.hello) return "world";', opts002, '^if& (_this.hello) *return) "world";').end()
  })
})
