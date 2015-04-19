var Q = require('../kew')

// create a deferred which returns a promise
exports.testDeferredResolve = function (test) {
  var val = "ok"
  var defer = Q.defer()

  defer.promise
    .then(function (data) {
      test.equal(data, val, "Promise successfully returned")
      test.done()
    })

  setTimeout(function () {
    defer.resolve(val)
  }, 50)
}

// make sure a deferred can only resolve once
exports.testDeferredResolveOnce = function (test) {
  var defer = Q.defer()

  try {
    defer.resolve(true)
    defer.resolve(true)
    test.fail("Unable to resolve the same deferred twice")
  } catch (e) {
  }

  test.done()
}

// create a deferred which returns a failed promise
exports.testDeferredReject = function (test) {
  var err = new Error("hello")
  var defer = Q.defer()

  defer.promise
    .fail(function (e) {
      test.equal(e, err, "Promise successfully failed")
      test.done()
    })

  setTimeout(function () {
    defer.reject(err)
  }, 50)
}

// make sure a deferred can only reject once
exports.testDeferredRejectOnce = function (test) {
  var defer = Q.defer()

  try {
    defer.reject(new Error("nope 1"))
    defer.reject(new Error("nope 2"))
    test.fail("Unable to reject the same deferred twice")
  } catch (e) {
  }

  test.done()
}

// make sure a deferred can only reject once
exports.testDeferAndRejectFail = function (test) {
  var defer

  try {
    defer = Q.defer()
    defer.reject(new Error("nope 1"))
    defer.resolve(true)
    test.fail("Unable to reject and resolve the same deferred")
  } catch (e) {
    test.ok(true, "Unable to reject and resolve same deferred")
  }

  try {
    defer = Q.defer()
    defer.resolve(true)
    defer.reject(new Error("nope 1"))
    test.fail("Unable to reject and resolve the same deferred")
  } catch (e) {
    test.ok(true, "Unable to reject and resolve same deferred")
  }

  test.done()
}

// create a deferred which resolves with a node-standard callback
exports.testDeferredResolverSuccess = function (test) {
  var val = "ok"
  var defer = Q.defer()
  var callback = defer.makeNodeResolver()

  defer.promise
    .then(function (data) {
      test.equal(data, val, "Promise successfully returned")
      test.done()
    })

  setTimeout(function () {
    callback(null, val)
  }, 50)
}

// create a deferred which rejects with a node-standard callback
exports.testDeferredResolverSuccess = function (test) {
  var err = new Error("hello")
  var defer = Q.defer()
  var callback = defer.makeNodeResolver()

  defer.promise
    .fail(function (e) {
      test.equal(e, err, "Promise successfully failed")
      test.done()
    })

  setTimeout(function () {
    callback(err)
  }, 50)
}