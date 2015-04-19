'use strict';
/*jshint asi: true*/

var test                =  require('tap').test
  , path                =  require('path')
  , fs                  =  require('fs')
  , hideSemicolonsTheme =  require('../themes/hide-semicolons')
  , home                =  path.join(__dirname, 'fixtures', 'home')
  , rcpath              =  path.join(home, '.cardinalrc')
  , existsSync          =  fs.existsSync || path.existsSync
  , settingsResolve     =  require.resolve('../settings')
  , settings

function setup () {
  delete require.cache[settingsResolve]
  settings = require(settingsResolve)
}

function writerc(config) {
  fs.writeFileSync(rcpath, JSON.stringify(config), 'utf-8')
}

function removerc () {
  fs.unlinkSync(rcpath)
}

function resolveTheme (config) {
  writerc(config)
  var result = settings.resolveTheme(home)
  removerc()
  return result;
}

function getSettings (config) {
  writerc(config)
  var result = settings.getSettings(home)
  removerc()
  return result;
}

if (!existsSync(home)) fs.mkdirSync(home);

test('no .cardinalrc in home', function (t) {
  setup()
  var theme = settings.resolveTheme(home)
  t.equals(theme, undefined, 'resolves no theme') 
  t.end()
})

test('.cardinalrc with theme "hide-semicolons" in home', function (t) {
  setup()
  var theme = resolveTheme({ theme: "hide-semicolons" })
  t.deepEquals(theme, hideSemicolonsTheme, 'resolves hide-semicolons theme') 
  t.end()
})

test('.cardinalrc with full path to "hide-semicolons.js" in home', function (t) {
  setup()
  var theme = resolveTheme({ theme: path.join(__dirname, '..', 'themes', 'hide-semicolons.js') })
  t.deepEquals(theme, hideSemicolonsTheme, 'resolves hide-semicolons theme') 
  t.end()
})

test('.cardinalrc with linenos: true', function (t) {
  setup()
  var opts = { linenos: true }
  t.deepEquals(getSettings(opts), opts)
  t.end()
})

test('.cardinalrc with linenos: true and theme', function (t) {
  setup()
  var opts = { linenos: true, theme: 'some theme' }
  t.deepEquals(getSettings(opts), opts)
  t.end()
})

