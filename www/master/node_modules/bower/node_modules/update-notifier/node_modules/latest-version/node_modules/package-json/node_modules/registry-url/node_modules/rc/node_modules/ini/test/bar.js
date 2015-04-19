//test that parse(stringify(obj) deepEqu

var ini = require('../')
var test = require('tap').test

var data = {
  'number':  {count: 10},
  'string':  {drink: 'white russian'},
  'boolean': {isTrue: true},
  'nested boolean': {theDude: {abides: true, rugCount: 1}}
}


test('parse(stringify(x)) deepEqual x', function (t) {

  for (var k in data) {
    var s = ini.stringify(data[k])
    console.log(s, data[k])
    t.deepEqual(ini.parse(s), data[k])
  }

  t.end() 
})
