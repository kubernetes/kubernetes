var Q = require('../kew')

exports.testThen = function (test) {
  var detectedScope = null
  Q.resolve(true).then(function () {
    detectedScope = this
  })
  test.ok(Q.isPromise(detectedScope), 'then() should be called in context of promise')
  test.done()
}

exports.testFail = function (test) {
  var detectedScope = null
  Q.reject(new Error()).fail(function () {
    detectedScope = this
  })
  test.ok(Q.isPromise(detectedScope), 'fail() should be called in context of promise')
  test.done()
}

exports.testThenBound = function (test) {
  var detectedScope = scope
  var scope = {}
  Q.resolve(true).thenBound(function () {
    detectedScope = scope
  }, scope)
  test.ok(detectedScope === scope, 'thenScoped() should be called in context of scope')
  test.done()
}

exports.testFailBound = function (test) {
  var detectedScope = scope
  var scope = {}
  Q.reject(new Error()).failBound(function () {
    detectedScope = scope
  }, scope)
  test.equal(detectedScope, scope, 'failBound() should be called in context of scope')
  test.done()
}

exports.testThenBoundWithArgs = function (test) {
  var detectedScope = scope
  var scope = {}
  Q.resolve(-1).thenBound(function (a, b, c, d) {
    test.equal(a, 1)
    test.equal(b, 2)
    test.equal(c, 3)
    test.equal(d, -1)
    detectedScope = scope
  }, scope, 1, 2, 3)
  test.ok(detectedScope === scope, 'failScoped() should be called in context of scope')
  test.done()
}

