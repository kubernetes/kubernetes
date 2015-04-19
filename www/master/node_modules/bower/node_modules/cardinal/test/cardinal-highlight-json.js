'use strict';
/*jshint asi: true*/

var test = require('tap').test
  , util = require('util')
  , customTheme = require('./fixtures/custom') 
  , cardinal = require('..')


function inspect (obj) {
  return console.log(util.inspect(obj, false, 5, false))
}

var json = JSON.stringify({
  foo: 'bar',
  baz: 'quux',
  bam: null
});

test('supplying custom theme', function (t) {
  var highlighted = cardinal.highlight(json, { json: true, theme: customTheme });

  t.equals(highlighted, '\u001b[33m{\u001b[39m\u001b[92m"foo"\u001b[39m\u001b[93m:\u001b[39m\u001b[92m"bar"\u001b[39m\u001b[32m,\u001b[39m\u001b[92m"baz"\u001b[39m\u001b[93m:\u001b[39m\u001b[92m"quux"\u001b[39m\u001b[32m,\u001b[39m\u001b[92m"bam"\u001b[39m\u001b[93m:\u001b[39m\u001b[90mnull\u001b[39m\u001b[33m}\u001b[39m')
  t.end();
});

test('not supplying custom theme', function (t) {
  var highlighted = cardinal.highlight(json, { json: true });

  t.equals(highlighted, '\u001b[33m{\u001b[39m\u001b[32m"foo"\u001b[39m\u001b[93m:\u001b[39m\u001b[92m"bar"\u001b[39m\u001b[32m,\u001b[39m\u001b[32m"baz"\u001b[39m\u001b[93m:\u001b[39m\u001b[92m"quux"\u001b[39m\u001b[32m,\u001b[39m\u001b[32m"bam"\u001b[39m\u001b[93m:\u001b[39m\u001b[90mnull\u001b[39m\u001b[33m}\u001b[39m')
  t.end();
});

test('without json option', function (t) {
  try {
    cardinal.highlight(json);
  } catch (e) {
    t.similar(e.message, /Unable to perform highlight. The code contained syntax errors.* Line 1: Unexpected token /)
    t.end();
  }
});
