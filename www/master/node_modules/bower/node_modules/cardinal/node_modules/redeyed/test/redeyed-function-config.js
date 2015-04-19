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

test('\nfunction config, keywords', function (t) {

  var opts001 = { Keyword: { _default: function (s) { return '*' + s + '&'; } } };
  t.test('\n# ' + inspect(opts001), function (t) {
    t.assertSurrounds('this', opts001, '*this&')

    t.assertSurrounds('this ', opts001, '*this& ')
    t.assertSurrounds(' this', opts001, ' *this&')
    t.assertSurrounds('  this  ', opts001, '  *this&  ')
    t.assertSurrounds('if (a == 1) return', opts001, '*if& (a == 1) *return&')
    t.assertSurrounds('var n = new Test();', opts001, '*var& n = *new& Test();')
    t.assertSurrounds(
        [ 'function foo (bar) {'
        , ' var a = 3;'
        , ' return bar + a;'
        , '}'
        ].join('\n')
      , opts001
      , [ '*function& foo (bar) {'
        , ' *var& a = 3;'
        , ' *return& bar + a;'
        , '}'
        ].join('\n'))
    t.end()
  })

  var opts002 = { 
    Keyword: { 
        'function': function (s) { return '^' + s + '&' }
      , 'return':  function (s) { return '(' + s + ')' }
      , _default: function (s) { return '*' + s + '&' }
    } 
  };  

  t.test('\n# ' + inspect(opts002), function (t) {
    t.assertSurrounds(
        [ 'function foo (bar) {'
        , ' var a = 3;'
        , ' return bar + a;'
        , '}'
        ].join('\n')
      , opts002
      , [ '^function& foo (bar) {'
        , ' *var& a = 3;'
        , ' (return) bar + a;'
        , '}'
        ].join('\n'))
    t.end()
  })
})

test('#\n functin config - resolving', function (t) {
  var opts001 = { 
      Keyword: { 
        'var': function (s) { return '^' + s + '&' }
      }
    , _default: function (s) { return '*' + s + '&' }
  };  
  t.test('\n# specific but no type default and root default - root default not applied' + inspect(opts001), function (t) {
    t.assertSurrounds('var n = new Test();', opts001, '^var& n = new Test();').end();
  })

  var opts002 = { 
      Keyword: { 
        'var': function (s) { return '^' + s + '&' }
      , _default: function (s) { return '*' + s + '&' }
      }
    , _default: function (s) { return '(' + s + ')' }
  };  
  t.test('\n# no type default but root default' + inspect(opts002), function (t) {
    t.assertSurrounds('var n = new Test();', opts002, '^var& n = *new& Test();').end();
  })
})

test('#\n function config - replacing', function (t) {
  var opts001 = { 
      Keyword: { 
        'var': function () { return 'const' }
      }
  };  
  t.test('\n# type default and root default (type wins)' + inspect(opts001), function (t) {
    t.assertSurrounds('var n = new Test();', opts001, 'const n = new Test();').end();
  })

  var opts002 = { 
      Keyword: { 
        _default: function () { return 'const' }
      }
  };  
  t.test('\n# type default' + inspect(opts002), function (t) {
    t.assertSurrounds('var n = new Test();', opts002, 'const n = const Test();').end();
  })
  
  var opts003 = { 
      Keyword: { 
          'new': function () { return 'NEW'; }
        , _default: function () { return 'const' }
      }
  };  
  t.test('\n# specific and type default' + inspect(opts003), function (t) {
    t.assertSurrounds('var n = new Test();', opts003, 'const n = NEW Test();').end();
  })

  var opts004 = { 
      Keyword: { 
        _default: function (s) { return s.toUpperCase() }
      }
      , _default: function (s) { return 'not applied'; }
  };  
  t.test('\n# type default and root default (type wins)' + inspect(opts004), function (t) {
    t.assertSurrounds('var n = new Test();', opts004, 'VAR n = NEW Test();').end();
  })

  var opts005 = { 
        Keyword: { }
      , _default: function (s) { return s.toUpperCase() }
  };  
  t.test('\n# no type default only root default - not applied' + inspect(opts005), function (t) {
    t.assertSurrounds('var n = new Test();', opts005, 'var n = new Test();').end();
  })
})
