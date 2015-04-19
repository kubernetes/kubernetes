var CC = require('../index.js').ConfigChain
var test = require('tap').test

var f1 = '/tmp/f1.ini'
var f2 = '/tmp/f2.json'

var ini = require('ini')

var f1data = {foo: {bar: 'baz'}, bloo: 'jaus'}
var f2data = {oof: {rab: 'zab'}, oolb: 'suaj'}

var fs = require('fs')

fs.writeFileSync(f1, ini.stringify(f1data), 'utf8')
fs.writeFileSync(f2, JSON.stringify(f2data), 'utf8')

test('test saving and loading ini files', function (t) {
  new CC()
    .add({grelb:'blerg'}, 'opt')
    .addFile(f1, 'ini', 'inifile')
    .addFile(f2, 'json', 'jsonfile')
    .on('load', function (cc) {

      t.same(cc.snapshot, { grelb: 'blerg',
                            bloo: 'jaus',
                            foo: { bar: 'baz' },
                            oof: { rab: 'zab' },
                            oolb: 'suaj' })

      t.same(cc.list, [ { grelb: 'blerg' },
                        { bloo: 'jaus', foo: { bar: 'baz' } },
                        { oof: { rab: 'zab' }, oolb: 'suaj' } ])

      cc.set('grelb', 'brelg', 'opt')
        .set('foo', 'zoo', 'inifile')
        .set('oof', 'ooz', 'jsonfile')
        .save('inifile')
        .save('jsonfile')
        .on('save', function () {
          t.equal(fs.readFileSync(f1, 'utf8'),
                  "bloo = jaus\nfoo = zoo\n")
          t.equal(fs.readFileSync(f2, 'utf8'),
                  "{\"oof\":\"ooz\",\"oolb\":\"suaj\"}")

          t.same(cc.snapshot, { grelb: 'brelg',
                                bloo: 'jaus',
                                foo: 'zoo',
                                oof: 'ooz',
                                oolb: 'suaj' })

          t.same(cc.list, [ { grelb: 'brelg' },
                            { bloo: 'jaus', foo: 'zoo' },
                            { oof: 'ooz', oolb: 'suaj' } ])

          t.pass('ok')
          t.end()
        })
    })
})
