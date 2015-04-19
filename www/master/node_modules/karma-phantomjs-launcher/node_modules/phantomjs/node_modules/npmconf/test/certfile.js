var test = require('tap').test
var npmconf = require('../npmconf.js')
var common = require('./00-setup.js')
var path = require('path')
var fs = require('fs')

test('cafile loads as ca', function (t) {
  var cafile = path.join(__dirname, 'fixtures', 'multi-ca')

  npmconf.load({cafile: cafile}, function (er, conf) {
    if (er) throw er

    t.same(conf.get('cafile'), cafile)
    t.same(conf.get('ca').join('\n'), fs.readFileSync(cafile, 'utf8').trim())
    t.end()
  })
})
