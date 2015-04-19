var cc = require('..')
var assert = require('assert')

assert.deepEqual({
  hello: true
}, cc.env('test_', {
  'test_hello': true,
  'ignore_this': 4,
  'ignore_test_this_too': []
}))
