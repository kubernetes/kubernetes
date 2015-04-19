var Q = require('../kew')
var originalQ = require('q')

// test that fin() works with a synchronous resolve
exports.testSynchronousThenAndFin = function (test) {
  var vals = ['a', 'b']
  var counter = 0

  var promise1 = Q.resolve(vals[0])
  var promise2 = promise1.fin(function () {
    counter++
  })
  var promise3 = promise2.then(function (data) {
    if (data === vals[0]) return vals[1]
  })
  var promise4 = promise3.fin(function () {
    counter++
  })

  Q.all([promise2, promise4])
    .then(function (data) {
      test.equal(counter, 2, "fin() should have been called twice")
      test.equal(data[0], vals[0], "first fin() should return the first val")
      test.equal(data[1], vals[1], "second fin() should return the second val")
      test.done()
    })
}

// test that fin() works with a synchronous reject
exports.testSynchronousFailAndFin = function (test) {
  var errs = []
  errs.push(new Error('nope 1'))
  errs.push(new Error('nope 2'))
  var counter = 0

  var promise1 = Q.reject(errs[0])
  var promise2 = promise1.fin(function () {
    counter++
  })
  var promise3 = promise2.fail(function (e) {
    if (e === errs[0]) throw errs[1]
  })
  var promise4 = promise3.fin(function () {
    counter++
  })

  Q.all([
    promise2.fail(function (e) {
      return e === errs[0]
    }),
    promise4.fail(function (e) {
      return e === errs[1]
    })
  ])
    .then(function (data) {
      test.equal(counter, 2, "fin() should have been called twice")
      test.equal(data[0] && data[1], true, "all promises should return true")
      test.done()
    })
}

// test that fin() works with an asynchrnous resolve
exports.testAsynchronousThenAndFin = function (test) {
  var vals = ['a', 'b']
  var counter = 0

  var defer = Q.defer()
  setTimeout(function () {
    defer.resolve(vals[0])
  })
  var promise1 = defer.promise
  var promise2 = promise1.fin(function () {
    counter++
  })
  var promise3 = promise2.then(function (data) {
    if (data !== vals[0]) return

    var defer = Q.defer()
    setTimeout(function () {
      defer.resolve(vals[1])
    })
    return defer.promise
  })
  var promise4 = promise3.fin(function () {
    counter++
  })

  Q.all([promise2, promise4])
    .then(function (data) {
      test.equal(counter, 2, "fin() should have been called twice")
      test.equal(data[0], vals[0], "first fin() should return the first val")
      test.equal(data[1], vals[1], "second fin() should return the second val")
      test.done()
    })
}

// test that fin() works with an asynchronous reject
exports.testAsynchronousFailAndFin = function (test) {
  var errs = []
  errs.push(new Error('nope 1'))
  errs.push(new Error('nope 2'))
  var counter = 0

  var defer = Q.defer()
  setTimeout(function () {
    defer.reject(errs[0])
  }, 10)
  var promise1 = defer.promise
  var promise2 = promise1.fin(function () {
    counter++
  })
  var promise3 = promise2.fail(function (e) {
    if (e !== errs[0]) return

    var defer = Q.defer()
    setTimeout(function () {
      defer.reject(errs[1])
    }, 10)

    return defer.promise
  })
  var promise4 = promise3.fin(function () {
    counter++
  })

  Q.all([
    promise2.fail(function (e) {
      return e === errs[0]
    }),
    promise4.fail(function (e) {
      return e === errs[1]
    })
  ])
    .then(function (data) {
      test.equal(counter, 2, "fin() should have been called twice")
      test.equal(data[0] && data[1], true, "all promises should return true")
      test.done()
    })
}

// test several thens chaining
exports.testChainedThens = function (test) {
  var promise1 = Q.resolve('a')
  var promise2 = promise1.then(function(data) {
    return data + 'b'
  })
  var promise3 = promise2.then(function (data) {
    return data + 'c'
  })
  // testing the same promise again to make sure they can run side by side
  var promise4 = promise2.then(function (data) {
    return data + 'c'
  })

  Q.all([promise1, promise2, promise3, promise4])
    .then(function (data) {
      test.equal(data[0], 'a')
      test.equal(data[1], 'ab')
      test.equal(data[2], 'abc')
      test.equal(data[3], 'abc')
      test.done()
    })
}

// test several fails chaining
exports.testChainedFails = function (test) {
  var errs = []
  errs.push(new Error("first err"))
  errs.push(new Error("second err"))
  errs.push(new Error("third err"))

  var promise1 = Q.reject(errs[0])
  var promise2 = promise1.fail(function (e) {
    if (e === errs[0]) throw errs[1]
  })
  var promise3 = promise2.fail(function (e) {
    if (e === errs[1]) throw errs[2]
  })
  var promise4 = promise2.fail(function (e) {
    if (e === errs[1]) throw errs[2]
  })

  Q.all([
    promise1.fail(function (e) {
      return e === errs[0]
    }),
    promise2.fail(function (e) {
      return e === errs[1]
    }),
    promise3.fail(function (e) {
      return e === errs[2]
    }),
    promise4.fail(function (e) {
      return e === errs[2]
    })
  ])
  .then(function (data) {
    test.equal(data[0] && data[1] && data[2] && data[3], true)
    test.done()
  })
}

// test that we can call end without callbacks and not fail
exports.testEndNoCallbacks = function (test) {
  Q.resolve(true).end()
  test.ok("Ended successfully")
  test.done()
}

// test that we can call end with callbacks and fail
exports.testEndNoCallbacksThrows = function (test) {
  var testError = new Error('Testing')
  try {
    Q.reject(testError).end()
    test.fail("Should throw an error")
  } catch (e) {
    test.equal(e, testError, "Should throw the correct error")
  }
  test.done()
}

// test chaining when a promise returns a promise
exports.testChainedPromises = function (test) {
  var err = new Error('nope')
  var val = 'ok'

  var shouldFail = Q.reject(err)
  var shouldSucceed = Q.resolve(val)

  Q.resolve("start")
    .then(function () {
      return shouldFail
    })
    .fail(function (e) {
      if (e === err) return shouldSucceed
      else throw e
    })
    .then(function (data) {
      test.equal(data, val, "val should be returned")
      test.done()
    })
}

// test .end() is called with no parent scope (causing an uncaught exception)
exports.testChainedEndUncaught = function (test) {
  var errs = []
  errs.push(new Error('nope 1'))
  errs.push(new Error('nope 2'))
  errs.push(new Error('nope 3'))

  process.on('uncaughtException', function (e) {
    test.equal(e, errs.shift(), "Error should be uncaught")
    if (errs.length === 0) test.done()
  })

  var defer = Q.defer()
  defer.promise.end()

  var promise1 = defer.promise
  var promise2 = promise1.fail(function (e) {
    if (e === errs[0]) throw errs[1]
  })
  var promise3 = promise2.fail(function (e) {
    if (e === errs[1]) throw errs[2]
  })

  promise1.end()
  promise2.end()
  promise3.end()

  setTimeout(function () {
    defer.reject(errs[0])
  }, 10)
}

// test .end() is called with a parent scope and is caught
exports.testChainedCaught = function (test) {
  var err = new Error('nope')

  try {
    Q.reject(err).end()
  } catch (e) {
    test.equal(e, err, "Error should be caught")
    test.done()
  }
}

// test a mix of fails and thens
exports.testChainedMixed = function (test) {
  var errs = []
  errs.push(new Error('nope 1'))
  errs.push(new Error('nope 2'))
  errs.push(new Error('nope 3'))

  var vals = [3, 2, 1]

  var promise1 = Q.reject(errs[0])
  var promise2 = promise1.fail(function (e) {
    if (e === errs[0]) return vals[0]
  })
  var promise3 = promise2.then(function (data) {
    if (data === vals[0]) throw errs[1]
  })
  var promise4 = promise3.fail(function (e) {
    if (e === errs[1]) return vals[1]
  })
  var promise5 = promise4.then(function (data) {
    if (data === vals[1]) throw errs[2]
  })
  var promise6 = promise5.fail(function (e) {
    if (e === errs[2]) return vals[2]
  })

  Q.all([
    promise1.fail(function (e) {
      return e === errs[0]
    }),
    promise2.then(function (data) {
      return data === vals[0]
    }),
    promise3.fail(function (e) {
      return e === errs[1]
    }),
    promise4.then(function (data) {
      return data === vals[1]
    }),
    promise5.fail(function (e) {
      return e === errs[2]
    }),
    promise6.then(function (data) {
      return data === vals[2]
    })
  ])
  .then(function (data) {
    test.equal(data[0] && data[1] && data[2] && data[3] && data[4] && data[5], true, "All values should return true")
    test.done()
  })
}

exports.testInteroperabilityWithOtherPromises = function(test) {
  var promise1 = Q.defer()
  promise1.then(function(value) {
    return originalQ(1 + value)
  }).then(function(result) {
    test.equal(result, 11)
  })

  var promise2 = Q.defer(),
      errToThrow = new Error('error')
  promise2.then(function() {
    return originalQ.reject(errToThrow)
  }).fail(function(err) {
    test.equal(err, errToThrow)
  })

  promise1.resolve(10)
  promise2.resolve()

  Q.all([promise1, promise2]).then(function() {
    test.done()
  })
}

exports.testAllSettled = function(test) {
  var promise1 = Q.resolve('woot')
  var promise2 = Q.reject(new Error('oops'))

  Q.allSettled([promise1, promise2, 'just a string'])
    .then(function (data) {
      test.equals('fulfilled', data[0].state)
      test.equals('woot', data[0].value)
      test.equals('rejected', data[1].state)
      test.equals('oops', data[1].reason.message)
      test.equals('fulfilled', data[2].state)
      test.equals('just a string', data[2].value)
    })

  Q.allSettled([])
    .then(function (data) {
      test.equals(0, data.length)
      test.done()
    })
}

exports.testTimeout = function(test) {
  var promise = Q.delay(50).timeout(45, 'Timeout message')
  promise.then(function () {
    test.fail('The promise is supposed to be timeout')
  })
  .fail(function (e) {
    test.equals('Timeout message', e.message, 'The error message should be the one passed into timeout()')
  })
  .fin(test.done)
}

exports.testNotTimeout = function(test) {
  var promise = Q.delay('expected data', 40).timeout(45, 'Timeout message')
  promise.then(function (data) {
    test.equals('expected data', data, 'The data should be the data from the original promise')
  })
  .fail(function (e) {
    test.fail('The promise is supposed to be resolved before the timeout')
  })
  .fin(test.done)
}

exports.testNotTimeoutButReject = function(test) {
  var promise = Q.delay(40).then(function() {throw new Error('Reject message')}).timeout(45, 'Timeout message')
  promise.then(function (data) {
    test.fail('The promise is supposed to be rejected')
  })
  .fail(function (e) {
    test.equals('Reject message', e.message, 'The error message should be from the original promise')
  })
  .fin(test.done)
}

exports.testDelay = function (test) {
  var timePassed = false
  setTimeout(function () {
    timePassed = true
  }, 10)
  Q.resolve('expected').delay(20).then(function (result) {
    test.equal('expected', result)
    test.ok(timePassed)
    test.done()
  })
}
