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

test('\nstring config, keywords', function (t) {

  var opts001 = { Keyword: { _default: '*:&' } };
  t.test('\n# ' + inspect(opts001), function (t) {
    t.assertSurrounds('this', opts001, '*this&')
    t.assertSurrounds('if (a == 1) return', opts001, '*if& (a == 1) *return&')
    t.assertSurrounds('var n = new Test();', opts001, '*var& n = *new& Test();')
    t.end()
  })

  var opts002 = { 
    Keyword: { 
        'function': '^:'
      , 'return':  '(:)' 
      , _default: '*:&'
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

test('\nstring configs resolve from type and root', function (t) {
  var code = 'var a = new Test();'
  
  function run(t, conf, expected, code_) {
    t.test('\n# '  + inspect(conf), function (t) {
      t.assertSurrounds(code_ || code, conf, expected);
      t.end()
    })
  }

  // at least the token kind has to be configured in order for the root_default to be applied
  // otherwise a root._default would affect all tokens, even the ones we want to leave unchanged
  run(t, { _default: '*:' }, 'var a = new Test();')

  t.test('\n\n# only before or after specified, but no root._default', function (t) {
    run(t, { Keyword: { _default: '*:' } }, '*var a = *new Test();')
    run(t, { Keyword: { _default: ':-' } }, 'var- a = new- Test();')
    t.end()
  })

  t.test('\n\n# resolve missing from root._default', function (t) {
    run(t, { Keyword: { _default: '*:' }, _default: '(:-' }, '*var- a = *new- Test();')
    run(t, { Keyword: { _default: ':-' }, _default: '*:)' }, '*var- a = *new- Test();')
    t.end()
  })

  t.test('\n\n# no resolve if all specified', function (t) {
    run(t, { Keyword: { _default: '+:-' }, _default: '*:)' }, '+var- a = +new- Test();')
    run(t, { Keyword: { _default: ':-' }, _default: ':)' }, 'var- a = new- Test();')
    t.end()
  })

  t.test('\n\n# resolve specific token no defaults', function (t) {
    run(t, { Keyword: { 'var': '*:' } }, '*var a = new Test();')
    run(t, { Keyword: { 'var': ':-' } }, 'var- a = new Test();')
    t.end()
  })

  t.test('\n\n# resolve specific token with type defaults', function (t) {
    run(t, { Keyword: { 'var': '*:', _default: ':-' } }, '*var- a = new- Test();')
    run(t, { Keyword: { 'var': '*:', _default: '(:-' } }, '*var- a = (new- Test();')
    run(t, { Keyword: { 'var': ':-', _default: '*:' } }, '*var- a = *new Test();')
    run(t, { Keyword: { 'var': ':-', _default: '*:)' } }, '*var- a = *new) Test();')
    run(t, { Keyword: { 'var': ':-', 'new': ':&', _default: '*:' } }, '*var- a = *new& Test();')
    t.end()
  })

  t.test(
      '\n\n# resolve specific token with root defaults, but no type default - root default not applied to unspecified tokens'
    , function (t) {
        run(t, { Keyword: { 'var': '*:' }, _default: ':-' }, '*var- a = new Test();')
        run(t, { Keyword: { 'var': ':-' }, _default: '*:' }, '*var- a = new Test();')
        t.end()
      }
  )

  t.test('\n\n# resolve specific token with type and root defaults', function (t) {
    run(t, { Keyword: { 'var': '*:', _default: '+:-' }, _default: ':)' }, '*var- a = +new- Test();')
    run(t, { Keyword: { 'var': ':-', _default: '*:+' }, _default: '(:' }, '*var- a = *new+ Test();')
    t.end()
  })

  t.test('all exact tokens undefined, but type default', function (t) {
    run(t, { 'Boolean': { 'true': undefined, 'false': undefined, _default: '+:-' } }, 'return +true- || +false-;', 'return true || false;')
  })
  
  t.end()
})
