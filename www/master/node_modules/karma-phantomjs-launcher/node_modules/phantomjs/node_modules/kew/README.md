kew: a lightweight (and super fast) promise/deferred framework for node.js
==================================

**kew** is a lightweight promise framework with an aim of providing a base set of functionality similar to that provided by the [Q library](https://github.com/kriskowal/q "Q").

A few answers (for a few questions)
-------

*Why'd we write it?*

During our initial usage of **Q** we found that it was consuming 80% of the cpu under load (primarily in chained database callbacks). We spent some time looking at patching **Q** and ultimately found that creating our own lightweight library for server-usage would suit our needs better than figuring out how to make a large cross-platform library more performant on one very specific platform.

*So this does everything Q does?*

Nope! **Q** is still an awesome library and does *way* more than **kew**. We support a tiny subset of the **Q** functionality (the subset that we happen to use in our actual use cases).

What are Promises?
-------

At its core, a *Promise* is a promise to return a value at some point in the future. A *Promise* represents a value that will be (or may return an error if something goes wrong). *Promises* heavily reduce the complexity of asynchronous coding in node.js-like environments. Example:

```javascript
// assuming the getUrlContent() function exists and retrieves the content of a url
var htmlPromise = getUrlContent(myUrl)

// we can then filter that through an http parser (our imaginary parseHtml() function) asynchronously (or maybe synchronously, who knows)
var tagsPromise = htmlPromise.then(parseHtml)

// and then filter it through another function (getLinks()) which retrieves only the link tags
var linksPromise = tagsPromise.then(getLinks)

// and then parses the actual urls from the links (using parseUrlsFromLinks())
var urlsPromise = linksPromise.then(parseUrlsFromLinks)

// finally, we have a promise that should only provide us with the urls and will run once all the previous steps have ran
urlsPromise.then(function (urls) {
  // do something with the urls
})
```

How do I use **kew**?
-------

As a precursor to all the examples, the following code must be at the top of your page:

```javascript
var Q = require('kew')
```

### Convert a literal into a promise

The easiest way to start a promise chain is by creating a new promise with a specified literal using Q.resolve() or Q.reject()

```javascript
// create a promise which passes a value to the next then() call
var successPromise = Q.resolve(val)

// create a promise which throws an error to be caught by the next fail() call
var failPromise = Q.reject(err)
```

In addition, you can create deferreds which can be used if you need to create a promise but resolve it later:

```javascript
// create the deferreds
var successDefer = Q.defer()
var failDefer = Q.defer()

// resolve or reject the defers in 1 second
setTimeout(function () {
  successDefer.resolve("ok")
  failDefer.reject(new Error("this failed"))
}, 1000)

// extract promises from the deferreds
var successPromise = successDefer.promise
var failPromise = failDefer.promise
```

If you have a node-style callback (taking an **Error** as the first parameter and a response as the second), you can call the magic `makeNodeResolver()` function on a defer to allow the defer to handle the callbacks:

```javascript
// create the deferred
var defer = Q.defer()

// some node-style function
getObjectFromDatabase(myObjectId, defer.makeNodeResolver())

// grab the output
defer.promise
  .then(function (obj) {
    // successfully retrieved the object
  })
  .fail(function (e) {
    // failed retrieving the object
  })
```

### Handling successful results with `.then()`

When a promise is resolved, you may call the `.then()` method to retrieve the value of the promise:

```javascript
promise.then(function (result) {
  // do something with the result here
})
```

`.then()` will in turn return a promise which will return the results of whatever it returns (asynchronously or not), allowing it to be chained indefinitely:

```javascript
Q.resolve('a')
  .then(function (result) {
    return result + 'b'
  })
  .then(function (result) {
    return result + 'c'
  })
  .then(function (result) {
    // result should be 'abc'
  })
```

In addition, `.then()` calls may return promises themselves, allowing for complex nesting of asynchronous calls in a flat manner:

```javascript
var htmlPromise = getUrlContent(myUrl)

var tagsPromise = htmlPromise.then(function (html) {
  if (!validHtml(html)) throw new Error("Invalid HTML")

  // pretend that parseHtml() returns a promise and is asynchronous
  return parseHtml(html)
})
```

### Handling errors with `.fail()`

If a promise is rejected for some reason, you may handle the failure case with the `.fail()` function:

```javascript
getObjectPromise
  .fail(function (e) {
    console.error("Failed to retrieve object", e)
  })
```

Like `.then()`, `.fail()` also returns a promise. If the `.fail()` call does not throw an error, it will pass the return value of the `.fail()` handler to any `.then()` calls chained to it:

```javascript
getObjectPromise
  .fail(function (e) {
    return retryGetObject(objId)
  })
  .then(function (obj) {
    // yay, we received an object
  })
  .fail(function (e) {
    // the retry failed :(
    console.error("Retrieving the object '" + objId + "' failed")
  })
})
```

If you've reached the end of your promise chain, you may call `.end()` which signifies that the promise chain is ended and any errors should be thrown in whatever scope the code is currently in:

```javascript
getObjectPromise
  // this will throw an error to the uncaught exception handler if the getObjectPromise call is asynchronous
  .end()
```

### `.fin()` when things are finished

You may attach a handler to a promise which will be ran regardless of whether the promise was resolved or rejected (but will only run upon completion). This is useful in the cases where you may have set up resources to run a request and wish to tear them down afterwards. `.fin()` will return the promise it is called upon:

```javascript
var connection = db.connect()

var itemPromise = db.getItem(itemId)
  .fin(function () {
    db.close()
  })
```

Other utility methods
-------

### `.all()` for many things

If you're waiting for multiple promises to return, you may pass them (mixed in with literals if you desire) into `.all()` which will create a promise that resolves successfully with an array of the results of the promises:

```javascript
var promises = []
promises.push(getUrlContent(url1))
promises.push(getUrlContent(url2))
promises.push(getUrlContent(url3))

Q.all(promises)
  .then(function (content) {
    // content[0] === content for url 1
    // content[1] === content for url 2
    // content[2] === content for url 3
  })
```

If any of the promises fail, Q.all will fail as well (so make sure to guard your promises with a `.fail()` call beforehand if you don't care whether they succeed or not):

```javascript
var promises = []
promises.push(getUrlContent(url1))
promises.push(getUrlContent(url2))
promises.push(getUrlContent(url3))

Q.all(promises)
  .fail(function (e) {
    console.log("Failed retrieving a url", e)
  })
```

### `.delay()` for future promises

If you need a little bit of delay (such as retrying a method call to a service that is "eventually consistent") before doing something else, ``Q.delay()`` is your friend:

```javascript
getUrlContent(url1)
.fail(function () {
  // Retry again after 200 milisseconds
  return Q.delay(200).then(function () {
    return getUrlContent(url1)
  })
})
```

If two arguments are passed, the first will be used as the return value, and the
second will be the delay in milliseconds.

```javascript
Q.delay(obj, 20).then(function (result) {
  console.log(result) // logs `obj` after 20ms
})
```

### `.fcall()` for delaying a function invocation until the next tick:
```javascript
// Assume someFn() is a synchronous 2 argument function you want to delay.
Q.fcall(someFn, arg1, arg2)
  .then(function (result) {
    console.log('someFn(' + arg1 + ', ' + arg2 + ') = ' + result)
  })
```

You can also use ``Q.fcall()`` with functions that return promises.

### `.nfcall()` for Node.js callbacks

``Q.nfcall()`` can be used to convert node-style callbacks into promises:

```javascript
Q.nfcall(fs.writeFile, '/tmp/myFile', 'content')
  .then(function () {
    console.log('File written successfully')
  })
  .fail(function (err) {
    console.log('Failed to write file', err)
  })
```


Contributing
------------

Questions, comments, bug reports, and pull requests are all welcome.
Submit them at [the project on GitHub](https://github.com/Obvious/kew/).

Bug reports that include steps-to-reproduce (including code) are the
best. Even better, make them in the form of pull requests that update
the test suite. Thanks!


Author
------

[Jeremy Stanley](https://github.com/azulus)
supported by
[The Obvious Corporation](http://obvious.com/).


License
-------

Copyright 2013 [The Obvious Corporation](http://obvious.com/).

Licensed under the Apache License, Version 2.0.
See the top-level file `LICENSE.TXT` and
(http://www.apache.org/licenses/LICENSE-2.0).
