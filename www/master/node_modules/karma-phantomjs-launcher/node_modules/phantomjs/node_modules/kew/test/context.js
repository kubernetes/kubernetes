var Q = require('../kew')

// test that contexts are propogated based on position
exports.testContextWithDelay = function (test) {

	Q.resolve(true)
	  .setContext({id: 1})
	  .then(function (val, context) {
	  	test.equal(context.id, 1, 'Should return the first context')
	  	return Q.delay(500)
	  })
	  .setContext({id: 2})
	  .then(function (val, context) {
	  	test.equal(context.id, 2, 'Should return the second context')
	  	return Q.delay(500)
	  })
	  .clearContext()
	  .then(function (val, context) {
	  	test.equal(typeof context, 'undefined', 'Should return an undefined context')
	  	return Q.delay(500)
	  })
	  .setContext({id: 3})
	  .fin(test.done)
}

// test adding and removing contexts
exports.testGeneralContextFlow = function (test) {
	Q.resolve(true)
		// test no context exists
	  .then(function (val, context) {
	  	test.equal(typeof context, 'undefined', 'Context should be undefined')
	  	throw new Error()
	  })
	  .fail(function (e, context) {
	  	test.equal(typeof context, 'undefined', 'Context should be undefined')
	  })

	  // set the context and mutate it
	  .setContext({counter: 1})
	  .then(function (val, context) {
	  	test.equal(context.counter, 1, 'Counter should be 1')
	  	context.counter++
	  })
	  .then(function (val, context) {
	  	test.equal(context.counter, 2, 'Counter should be 2')
	  	context.counter++
	  	throw new Error()
	  })
	  .fail(function (e, context) {
	  	test.equal(context.counter, 3, 'Counter should be 3')
	  })

	  // return a context
	  .then(function (val, context) {
	  	return Q.resolve(false)
	  	  .setContext({counter: 0})
	  })
	  .then(function (val, context) {
	  	test.equal(context.counter, 0, 'Counter should be 0')
	  	throw new Error()
	  })
	  .fail(function (e, context) {
	  	test.equal(context.counter, 0, 'Counter should be 0')
	  })

	  // returning a promise with a cleared context won't clear the parent context
	  .then(function (val, context) {
	  	return Q.resolve(false).clearContext()
	  })
	  .then(function (val, context) {
	  	test.equal(context.counter, 0, 'Counter should be 0')
	  	throw new Error()
	  })
	  .fail(function (e, context) {
	  	test.equal(context.counter, 0, 'Counter should be 0')
	  })

	  // test that clearing the context works
	  .clearContext()
	  .then(function (val, context) {
	  	test.equal(typeof context, 'undefined', 'Context should be undefined')
	  	throw new Error()
	  })
	  .fail(function (e, context) {
	  	test.equal(typeof context, 'undefined', 'Context should be undefined')
	  })

	  .fin(test.done)
}