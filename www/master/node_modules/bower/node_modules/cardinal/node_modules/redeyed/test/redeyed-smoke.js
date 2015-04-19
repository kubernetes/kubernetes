'use strict';
/*jshint asi: true*/

// applying redeyed to a bunch of files of contained libraries as a smoke test
var test     =  require('tap').test
  , path     =  require('path')
  , fs       =  require('fs')
  , readdirp =  require('readdirp')
  , redeyed  =  require('..')
  , node_modules =  path.join(__dirname, '..', 'node_modules')
  , tapdir       =  path.join(node_modules, 'tap')
  , esprimadir   =  path.join(node_modules, 'esprima')

test('tap', function (t) {
  var invalidTapFiles = [
      'async-map-ordered.js'
    , 'prof.js'
  ]

  readdirp({ root: tapdir, fileFilter: '*.js' })
    .on('data', function (entry) {
      
      if (~invalidTapFiles.indexOf(entry.name)) return

      var code = fs.readFileSync(entry.fullPath, 'utf-8')
        , result = redeyed(code, { Keyword: { 'var': '+:-' } }).code

      t.assert(~result.indexOf('+var-') || !(~result.indexOf('var ')), 'redeyed ' + entry.path)
    })
    .on('end', t.end.bind(t))
})

test('esprima', function (t) {

  readdirp({ root: esprimadir, fileFilter: '*.js' })
    .on('data', function (entry) {
      
      var code = fs.readFileSync(entry.fullPath, 'utf-8')
        , result = redeyed(code, { Keyword: { 'var': '+:-' } }).code

      t.assert(~result.indexOf('+var-') || !(~result.indexOf('var ')), 'redeyed ' + entry.path)
    })
    .on('end', t.end.bind(t))
})

test('redeyed', function (t) {

  readdirp({ root: path.join(__dirname, '..'), fileFilter: '*.js', directoryFilter: ['!.git', '!node_modules' ] })
    .on('data', function (entry) {
      
      var code = fs.readFileSync(entry.fullPath, 'utf-8')
        , result = redeyed(code, { Keyword: { 'var': '+:-' } }).code

        t.assert(~result.indexOf('+var-') || !(~result.indexOf('var ')), 'redeyed ' + entry.path)
    })
    .on('end', t.end.bind(t))
})
