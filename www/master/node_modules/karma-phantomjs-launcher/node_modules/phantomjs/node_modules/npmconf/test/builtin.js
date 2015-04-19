var test = require('tap').test
var npmconf = require('../npmconf.js')
var common = require('./00-setup.js')
var path = require('path')

var ucData = common.ucData

var envData = common.envData
var envDataFix = common.envDataFix

var gcData = { 'package-config:foo': 'boo' }

var biData = { 'builtin-config': true }

var cli = { foo: 'bar', heading: 'foo', 'git-tag-version': false }

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
  builtin:
    { path: common.builtin,
      type: 'ini',
      data: biData }
}

test('with builtin', function (t) {
  npmconf.load(cli, common.builtin, function (er, conf) {
    if (er) throw er
    t.same(conf.list, expectList)
    t.same(conf.sources, expectSources)
    t.same(npmconf.rootConf.list, [])
    t.equal(npmconf.rootConf.root, npmconf.defs.defaults)
    t.equal(conf.root, npmconf.defs.defaults)
    t.equal(conf.get('heading'), 'foo')
    t.equal(conf.get('git-tag-version'), false)
    t.end()
  })
})
