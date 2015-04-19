'use strict';
/*jshint asi: true*/

var test = require('tap').test
  , fs = require('fs')
  , customTheme = require('./fixtures/custom') 
  , cardinal = require('..')

function inspect(obj, depth) {
  console.log(require('util').inspect(obj, false, depth || 5, true));
}

test('\nhighlighting a block comment without line numbers', function (t) {
  var code = fs.readFileSync(__dirname + '/fixtures/block-comment.js', 'utf8');
  var highlighted = cardinal.highlight(code, { theme: customTheme });
  t.equal(highlighted, '\n\u001b[90m/**\n * This is a meaningless block jsdoc for a meaningless function.\n * Joins two strings, separating them to appear on two lines.\n * \n * @name foo\n * @function\n * @param uno {String} first string\n * @param dos {String} second string\n * @return {String} result of the join\n */\u001b[39m\n\u001b[96mmodule\u001b[39m\u001b[32m.\u001b[39m\u001b[96mexports\u001b[39m \u001b[93m=\u001b[39m \u001b[94mfunction\u001b[39m \u001b[96mfoo\u001b[39m \u001b[90m(\u001b[39m\u001b[96muno\u001b[39m\u001b[32m,\u001b[39m \u001b[96mdos\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m\n  \u001b[31mreturn\u001b[39m \u001b[96muno\u001b[39m \u001b[93m+\u001b[39m \u001b[92m\'\\n\'\u001b[39m \u001b[93m+\u001b[39m \u001b[96mdos\u001b[39m\u001b[90m;\u001b[39m\n\u001b[33m}\u001b[39m\n')
  t.end()
})

test('\nhighlighting a block comment with line numbers', function (t) {
  var code = fs.readFileSync(__dirname + '/fixtures/block-comment.js', 'utf8');
  var highlighted = cardinal.highlight(code, { theme: customTheme, linenos: true });
  t.equal(highlighted, '\u001b[90m 1: \n\u001b[90m 2: \u001b[90m/**\n\u001b[90m 3:  * This is a meaningless block jsdoc for a meaningless function.\n\u001b[90m 4:  * Joins two strings, separating them to appear on two lines.\n\u001b[90m 5:  * \n\u001b[90m 6:  * @name foo\n\u001b[90m 7:  * @function\n\u001b[90m 8:  * @param uno {String} first string\n\u001b[90m 9:  * @param dos {String} second string\n\u001b[90m10:  * @return {String} result of the join\n\u001b[90m11:  */\u001b[39m\n\u001b[90m12: \u001b[96mmodule\u001b[39m\u001b[32m.\u001b[39m\u001b[96mexports\u001b[39m \u001b[93m=\u001b[39m \u001b[94mfunction\u001b[39m \u001b[96mfoo\u001b[39m \u001b[90m(\u001b[39m\u001b[96muno\u001b[39m\u001b[32m,\u001b[39m \u001b[96mdos\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m\n\u001b[90m13:   \u001b[31mreturn\u001b[39m \u001b[96muno\u001b[39m \u001b[93m+\u001b[39m \u001b[92m\'\\n\'\u001b[39m \u001b[93m+\u001b[39m \u001b[96mdos\u001b[39m\u001b[90m;\u001b[39m\n\u001b[90m14: \u001b[33m}\u001b[39m')
  t.end()
})
