'use strict';
/*jshint asi: true*/

var test     =  require('tap').test
  , path     =  require('path')
  , fs       =  require('fs')
  , themesdir = path.join(__dirname, '..', 'themes')
  , allFiles = fs.readdirSync(themesdir)

test('validate themes by requiring all of them', function (t) {
  allFiles
    .filter(function (file) { return path.extname(file) === '.js'; })
    .forEach(function (theme) {
      try {
        t.ok(require(path.join(themesdir, theme)), theme + ' is valid')   
      } catch (e) {
        t.fail('theme: ' + theme + ' is invalid! ' + e.message)
      }
    })
  t.end()
})
  
