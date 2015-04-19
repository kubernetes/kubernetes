'use strict'
/*jshint asi: true, browser: true*/
/*global define window */

var test = require('tap').test
  , util = require('util')
  , redeyedExport = require('..')
  , redeyedkey = require.resolve('..')
  , esprima = require('esprima')

function setup() {
  // remove redeyed from require cache to force re-require for each test
  delete require.cache[redeyedkey];
  
  // remove globals
  delete global.window;
  delete global.define;
}

// TODO: need to run in vm in order to properly simulate require and module not being present
return;
test('define and window exist', function (t) {
  var defineCb
    , deps

  setup()  

  // declare browser globals
  global.window = { }

  global.define = function (deps_, cb) { 
    deps_ = deps 
    defineCb = cb 
  }

  define.amd = true

  var redeyed = require('..')
    , definedredeyed = defineCb(esprima)

  t.equal(window.redeyed, undefined, 'redeyed is not attached to window')
  t.notEqual(redeyed.toString(), redeyedExport.toString(), 'redeyed is not exported')
  t.equal(definedredeyed.toString(), redeyedExport.toString(), 'redeyed is defined')

  t.end()
})

test('window exists, but define doesn\'t', function (t) {
  setup()  

  // declare browser globals
  global.window = { esprima: esprima }
    
  var redeyed = require('..')

  t.equal(window.redeyed.toString(), redeyedExport.toString(), 'redeyed is attached to window')
  t.notEqual(redeyed.toString(), redeyedExport.toString(), 'redeyed is not exported')
  t.end()
})

test('neither window nor define exist', function (t) {
  setup()  

  var redeyed = require('..')

  t.equal(redeyed.toString(), redeyedExport.toString(), 'redeyed is exported')
  t.end()
})

