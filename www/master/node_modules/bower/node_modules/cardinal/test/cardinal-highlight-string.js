'use strict';
/*jshint asi: true*/

var test = require('tap').test
  , util = require('util')
  , fs = require('fs')
  , customTheme = require('./fixtures/custom') 
  , cardinal = require('..')

function inspect (obj) {
  return console.log(util.inspect(obj, false, 5, false))
}

var code = 'function foo() { var a = 3; return a > 2 ? true : false; }'
  , codeWithErrors = 'function () { var a = 3; return a > 2 ? true : false; }';

test('supplying custom theme', function (t) {
  var highlighted = cardinal.highlight(code, { theme: customTheme });

  t.equals(highlighted, '\u001b[94mfunction\u001b[39m \u001b[96mfoo\u001b[39m\u001b[90m(\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m \u001b[32mvar\u001b[39m \u001b[96ma\u001b[39m \u001b[93m=\u001b[39m \u001b[34m3\u001b[39m\u001b[90m;\u001b[39m \u001b[31mreturn\u001b[39m \u001b[96ma\u001b[39m \u001b[93m>\u001b[39m \u001b[34m2\u001b[39m \u001b[93m?\u001b[39m \u001b[31mtrue\u001b[39m \u001b[93m:\u001b[39m \u001b[91mfalse\u001b[39m\u001b[90m;\u001b[39m \u001b[33m}\u001b[39m')
  t.end()
})

test('not supplying custom theme', function (t) {
  var highlighted = cardinal.highlight(code);

  t.equals(highlighted, '\u001b[94mfunction\u001b[39m \u001b[37mfoo\u001b[39m\u001b[90m(\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m \u001b[32mvar\u001b[39m \u001b[37ma\u001b[39m \u001b[93m=\u001b[39m \u001b[34m3\u001b[39m\u001b[90m;\u001b[39m \u001b[31mreturn\u001b[39m \u001b[37ma\u001b[39m \u001b[93m>\u001b[39m \u001b[34m2\u001b[39m \u001b[93m?\u001b[39m \u001b[91mtrue\u001b[39m \u001b[93m:\u001b[39m \u001b[91mfalse\u001b[39m\u001b[90m;\u001b[39m \u001b[33m}\u001b[39m')
  t.end()
})

test('errornous code', function (t) {
  try {
    cardinal.highlight(codeWithErrors);
  } catch (e) {
    t.similar(e.message, /Unable to perform highlight. The code contained syntax errors.* Line 1: Unexpected token [(]/)
    t.end()
  }
})

test('line numbers no firstline given', function (t) {
  var highlighted = cardinal.highlight(code, { linenos: true });
  t.equals(highlighted, '\u001b[90m1: \u001b[94mfunction\u001b[39m \u001b[37mfoo\u001b[39m\u001b[90m(\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m \u001b[32mvar\u001b[39m \u001b[37ma\u001b[39m \u001b[93m=\u001b[39m \u001b[34m3\u001b[39m\u001b[90m;\u001b[39m \u001b[31mreturn\u001b[39m \u001b[37ma\u001b[39m \u001b[93m>\u001b[39m \u001b[34m2\u001b[39m \u001b[93m?\u001b[39m \u001b[91mtrue\u001b[39m \u001b[93m:\u001b[39m \u001b[91mfalse\u001b[39m\u001b[90m;\u001b[39m \u001b[33m}\u001b[39m')
  t.end()
})

test('line numbers firstline 99', function (t) {
  var highlighted = cardinal.highlight(code, { linenos: true, firstline: 99 });
  t.equals(highlighted, '\u001b[90m99: \u001b[94mfunction\u001b[39m \u001b[37mfoo\u001b[39m\u001b[90m(\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m \u001b[32mvar\u001b[39m \u001b[37ma\u001b[39m \u001b[93m=\u001b[39m \u001b[34m3\u001b[39m\u001b[90m;\u001b[39m \u001b[31mreturn\u001b[39m \u001b[37ma\u001b[39m \u001b[93m>\u001b[39m \u001b[34m2\u001b[39m \u001b[93m?\u001b[39m \u001b[91mtrue\u001b[39m \u001b[93m:\u001b[39m \u001b[91mfalse\u001b[39m\u001b[90m;\u001b[39m \u001b[33m}\u001b[39m')
  t.end()
})

test('line numbers multi line no first line given', function (t) {
  var multilineCode = '' + 
    function foo () {
      return 1;
    };
  var highlighted = cardinal.highlight(multilineCode, { linenos: true });
  t.equals(highlighted,'\u001b[90m1: \u001b[94mfunction\u001b[39m \u001b[37mfoo\u001b[39m\u001b[90m(\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m\n\u001b[90m2:       \u001b[31mreturn\u001b[39m \u001b[34m1\u001b[39m\u001b[90m;\u001b[39m\n\u001b[90m3:     \u001b[33m}\u001b[39m')
  t.end()
})

test('line numbers multi line first line 99', function (t) {
  var multilineCode = '' + 
    function foo () {
      return 1;
    };
  var highlighted = cardinal.highlight(multilineCode, { linenos: true, firstline: 99 });
  t.equals(highlighted,'\u001b[90m 99: \u001b[94mfunction\u001b[39m \u001b[37mfoo\u001b[39m\u001b[90m(\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m\n\u001b[90m100:       \u001b[31mreturn\u001b[39m \u001b[34m1\u001b[39m\u001b[90m;\u001b[39m\n\u001b[90m101:     \u001b[33m}\u001b[39m')
  t.end()
})
