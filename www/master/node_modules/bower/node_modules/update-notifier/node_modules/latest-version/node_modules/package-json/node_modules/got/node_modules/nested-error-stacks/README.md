Nested stacktraces for Node.js!
===============================

[![Build Status](https://travis-ci.org/mdlavin/nested-error-stacks.svg)](https://travis-ci.org/mdlavin/nested-error-stacks)
[![NPM version](https://badge.fury.io/js/nested-error-stacks.svg)](http://badge.fury.io/js/nested-error-stacks)
[![Dependency Status](https://david-dm.org/mdlavin/nested-error-stacks.svg)](https://david-dm.org/mdlavin/nested-error-stacks)

With this module, you can wrap a caught exception with extra context
for better debugging.  For example, a network error's stack would normally look
like this:

    Error: connect ECONNREFUSED
        at errnoException (net.js:904:11)
        at Object.afterConnect [as oncomplete] (net.js:895:19)

Using this module, you can wrap the Error with more context to get a stack
that looks like this:

    NestedError: Failed to communicate with localhost:8080
        at Socket.<anonymous> (/Users/mattlavin/Projects/nested-stacks/demo.js:6:18)
        at Socket.EventEmitter.emit (events.js:95:17)
        at net.js:440:14
        at process._tickCallback (node.js:415:13)
    Caused By: Error: connect ECONNREFUSED
        at errnoException (net.js:904:11)
        at Object.afterConnect [as oncomplete] (net.js:895:19)

How to wrap errors
==================

Here is an example program that uses this module to add more context to errors:

```javascript
    var NestedError = require('nested-error-stacks');
    var net = require('net');
    
    var client = net.connect({port: 8080});
    client.on('error', function (err) {
    var newErr = new NestedError("Failed to communicate with localhost:8080", err);
        console.log(newErr.stack);
    });
```
