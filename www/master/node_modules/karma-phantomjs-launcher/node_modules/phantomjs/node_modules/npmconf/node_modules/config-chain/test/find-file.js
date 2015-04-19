
var fs = require('fs')
  , assert = require('assert')
  , objx = {
    rand: Math.random()
  }

fs.writeFileSync('/tmp/random-test-config.json', JSON.stringify(objx))

var cc = require('../')
var path = cc.find('tmp/random-test-config.json')

assert.equal(path, '/tmp/random-test-config.json')