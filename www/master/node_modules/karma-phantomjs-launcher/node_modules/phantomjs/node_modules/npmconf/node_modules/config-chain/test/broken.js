

var cc = require('..')
var assert = require('assert')


//throw on invalid json
assert.throws(function () {
  cc(__dirname + '/broken.json')
})
