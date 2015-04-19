'use strict';
/*jshint asi: true*/

var test = require('tap').test
  , util = require('util')
  , redeyed = require('..')
  , esprima = require('esprima')

function inspect (obj) {
  return util.inspect(obj, false, 5, true)
}

test('redeyed result has esprima ast, tokens, comments and splits and transformed code', function (t) {
  var code = '// a comment\nvar a = 3;'
    , conf = { Keyword: { _default: '_:-' } }

    , ast    =  esprima.parse(code, { tokens: true, comment: true, range: true, tolerant: true })
    , tokens =  ast.tokens
    , comments = ast.comments

    , result = redeyed(code, conf)

    console.log(ast)
  t.deepEquals(result.ast, ast, 'ast')
  t.deepEquals(result.tokens, tokens, 'tokens')
  t.deepEquals(result.comments, comments, 'comments')
  t.notEquals(result.code, undefined, 'code')

  t.end()
});

test('redeyed result - { nojoin } has esprima ast, tokens, comments and splits but no transformed code', function (t) {
  var code = '// a comment\nvar a = 3;'
    , conf = { Keyword: { _default: '_:-' } }

    , ast    =  esprima.parse(code, { tokens: true, comment: true, range: true, tolerant: true })
    , tokens =  ast.tokens
    , comments = ast.comments

    , result = redeyed(code, conf, { nojoin: true })

  t.deepEquals(result.ast, ast, 'ast')
  t.deepEquals(result.tokens, tokens, 'tokens')
  t.deepEquals(result.comments, comments, 'comments')
  t.equals(result.code, undefined, 'code')

  t.end()
});
