var test = require('tap').test
var CC = require('../index.js').ConfigChain

var env = { foo_blaz : 'blzaa', foo_env : 'myenv' }
var jsonObj = { blaz: 'json', json: true }
var iniObj = { 'x.y.z': 'xyz', blaz: 'ini' }

var fs = require('fs')
var ini = require('ini')

fs.writeFileSync('/tmp/config-chain-class.json', JSON.stringify(jsonObj))
fs.writeFileSync('/tmp/config-chain-class.ini', ini.stringify(iniObj))

var http = require('http')
var reqs = 0
http.createServer(function (q, s) {
  if (++reqs === 2) this.close()
  if (q.url === '/json') {
    // make sure that the requests come back from the server
    // out of order.  they should still be ordered properly
    // in the resulting config object set.
    setTimeout(function () {
      s.setHeader('content-type', 'application/json')
      s.end(JSON.stringify({
        blaz: 'http',
        http: true,
        json: true
      }))
    }, 200)
  } else {
    s.setHeader('content-type', 'application/ini')
    s.end(ini.stringify({
      blaz: 'http',
      http: true,
      ini: true,
      json: false
    }))
  }
}).listen(1337)

test('basic class test', function (t) {
  var cc = new CC()
  var expectlist =
      [ { blaz: 'json', json: true },
        { 'x.y.z': 'xyz', blaz: 'ini' },
        { blaz: 'blzaa', env: 'myenv' },
        { blaz: 'http', http: true, json: true },
        { blaz: 'http', http: true, ini: true, json: false } ]

  cc.addFile('/tmp/config-chain-class.json')
    .addFile('/tmp/config-chain-class.ini')
    .addEnv('foo_', env)
    .addUrl('http://localhost:1337/json')
    .addUrl('http://localhost:1337/ini')
    .on('load', function () {
      t.same(cc.list, expectlist)
      t.same(cc.snapshot, { blaz: 'json',
                            json: true,
                            'x.y.z': 'xyz',
                            env: 'myenv',
                            http: true,
                            ini: true })

      cc.del('blaz', '/tmp/config-chain-class.json')
      t.same(cc.snapshot, { blaz: 'ini',
                            json: true,
                            'x.y.z': 'xyz',
                            env: 'myenv',
                            http: true,
                            ini: true })
      cc.del('blaz')
      t.same(cc.snapshot, { json: true,
                            'x.y.z': 'xyz',
                            env: 'myenv',
                            http: true,
                            ini: true })
      cc.shift()
      t.same(cc.snapshot, { 'x.y.z': 'xyz',
                            env: 'myenv',
                            http: true,
                            json: true,
                            ini: true })
      cc.shift()
      t.same(cc.snapshot, { env: 'myenv',
                            http: true,
                            json: true,
                            ini: true })
      cc.shift()
      t.same(cc.snapshot, { http: true,
                            json: true,
                            ini: true })
      cc.shift()
      t.same(cc.snapshot, { http: true,
                            ini: true,
                            json: false })
      cc.shift()
      t.same(cc.snapshot, {})
      t.end()
    })
})
