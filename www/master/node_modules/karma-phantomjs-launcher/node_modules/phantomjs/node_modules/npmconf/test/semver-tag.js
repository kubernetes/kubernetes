var test = require('tap').test
var npmconf = require('../npmconf.js')
var common = require('./00-setup.js')
var path = require('path')

var ucData = common.ucData

var envData = common.envData
var envDataFix = common.envDataFix

var gcData = { 'package-config:foo': 'boo' }

var biData = { 'builtin-config': true }

var cli = { tag: 'v2.x' }

var projectData = {}

var expectList =
[ cli,
  envDataFix,
  projectData,
  ucData,
  gcData,
  biData ]


var expectSources =
{ cli: { data: cli },
  env:
   { data: envDataFix,
     source: envData,
     prefix: '' },
  project:
    { path: path.resolve(__dirname, '..', '.npmrc'),
      type: 'ini',
      data: projectData },
  user:
   { path: common.userconfig,
     type: 'ini',
     data: ucData },
  global:
   { path: common.globalconfig,
     type: 'ini',
     data: gcData },
  builtin: { data: biData } }

test('tag cannot be a SemVer', function (t) {
  var messages = []
  console.warn = function (m) {
    messages.push(m)
  }

  var expect = [
    'invalid config tag="v2.x"',
    'invalid config Tag must not be a SemVer range'
  ]

  npmconf.load(cli, common.builtin, function (er, conf) {
    if (er) throw er
    t.equal(conf.get('tag'), 'latest')
    t.same(messages, expect)
    t.end()
  })
})
