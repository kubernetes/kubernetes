'use strict';
/*jshint asi: true*/

var test = require('tap').test
  , util = require('util')
  , fs = require('fs')
  , path = require('path')
  , customTheme = require('./fixtures/custom') 
  , cardinal = require('..')

function inspect (obj) {
  return console.log(util.inspect(obj, false, 5, false))
}

var file = path.join(__dirname, 'fixtures/foo.js')
  , fileWithErrors = path.join(__dirname, 'fixtures/foo-with-errors.js')

test('supplying custom theme', function (t) {
  cardinal.highlightFile(file, { theme: customTheme }, function (err, highlighted) {

    t.equals(null, err, 'no error')
    t.equals(highlighted, '\u001b[94mfunction\u001b[39m \u001b[96mfoo\u001b[39m\u001b[90m(\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m \n  \u001b[32mvar\u001b[39m \u001b[96ma\u001b[39m \u001b[93m=\u001b[39m \u001b[34m3\u001b[39m\u001b[90m;\u001b[39m \u001b[31mreturn\u001b[39m \u001b[96ma\u001b[39m \u001b[93m>\u001b[39m \u001b[34m2\u001b[39m \u001b[93m?\u001b[39m \u001b[31mtrue\u001b[39m \u001b[93m:\u001b[39m \u001b[91mfalse\u001b[39m\u001b[90m;\u001b[39m \n\u001b[33m}\u001b[39m\n')
    t.end()
  })
})

test('not supplying custom theme', function (t) {
  cardinal.highlightFile(file, function (err, highlighted) {

    t.equals(null, err, 'no error')
    t.equals(highlighted, '\u001b[94mfunction\u001b[39m \u001b[37mfoo\u001b[39m\u001b[90m(\u001b[39m\u001b[90m)\u001b[39m \u001b[33m{\u001b[39m \n  \u001b[32mvar\u001b[39m \u001b[37ma\u001b[39m \u001b[93m=\u001b[39m \u001b[34m3\u001b[39m\u001b[90m;\u001b[39m \u001b[31mreturn\u001b[39m \u001b[37ma\u001b[39m \u001b[93m>\u001b[39m \u001b[34m2\u001b[39m \u001b[93m?\u001b[39m \u001b[91mtrue\u001b[39m \u001b[93m:\u001b[39m \u001b[91mfalse\u001b[39m\u001b[90m;\u001b[39m \n\u001b[33m}\u001b[39m\n')
    t.end()
  })
})

test('errornous code', function (t) {
  cardinal.highlightFile(fileWithErrors, function (err, highlighted) {
    t.similar(err.message, /Unable to perform highlight. The code contained syntax errors.* Line 1: Unexpected token [(]/)
    t.end()
  })
})

test('non existing file', function (t) {
  cardinal.highlightFile('./not/existing', function (err, highlighted) {
    t.similar(err.message, /ENOENT, .*not.existing/)
    t.end()
  })
})
