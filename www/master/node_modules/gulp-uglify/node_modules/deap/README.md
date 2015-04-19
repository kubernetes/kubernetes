[![Build Status](https://travis-ci.org/selfcontained/deap.svg?branch=master)](https://travis-ci.org/selfcontained/deap)
[![Coverage Status](https://img.shields.io/coveralls/selfcontained/deap.svg)](https://coveralls.io/r/selfcontained/deap?branch=master)

===

[![browser support](https://ci.testling.com/selfcontained/deap.png)](http://ci.testling.com/selfcontained/deap)

deap
====

extend and merge objects, deep or shallow, in javascript


### installation

```bash
npm install deap
```

```javascript
var deap = require('deap');
```

### browser usage

**deap** assumes es5, so we recommend using an es5 shim for older browsers.  [Browserify](https://github.com/substack/node-browserify) is also recommended as a means to use this module client-side, but other module loaders for browsers will work with **deap** as well if you shim it.

### available functions

+ deap() and deap.extend() - **deep extend**
+ deap.merge() - **deep merge**
+ deap.update() - **deep update**
+ deap.extendShallow() - **shallow extend**
+ deap.mergeShallow() - **shallow merge**
+ deap.updateShallow() - **shallow update**
+ deap.clone() - **deep clone**

---

### deap() and deap.extend()

Deep extend.  Copy all the properties from one object onto another, cloning objects deeply.

Takes *n* number of arguments, modifies the first argument and returns it.

```javascript
var a = { name: 'Joe' };

deap.extend(a, { age: 26 }); // returns: a => { name: 'Joe', age: 26 }
deap.extend({}, someObj); // clone someObj
```

### deap.merge()

Deep merge.  Copy properties from one object to another, not replacing existing properties.

Takes *n* number of arguments, modifies the first argument and returns it.

```javascript
var a = { name: 'Joe', address: { number: 1234 };
deap.merge(a, { name: 'Jack', age: 26, phone: '555-555-5555', address: { number: 4321, street: 'University Blvd' });
// returns: a => { name: 'Joe', age: 26, phone: '555-555-5555', address: { number: 1234, street: 'University Blvd' }}
```

### deap.update()

Deep update.  Fill an object's existing properties from another object.

Takes *n* number of arguments, modifies the first argument and returns it.

```javascript
var a = { name: 'Joe', phone: '' };
deap.update(a, { age: 26, phone: '555-555-5555' }); // returns: a => { name: 'Joe', phone: '555-555-5555' }
```

---

## shallow only

If you prefer a shallow-only instance of **deap** you can require it specifically

```javascript
var deap = require('deap/shallow');

deap() && deap.extend(); // shallow extend
deap.merge(); //shallow merge
deap.update(); //shallow update
deap.clone(); // deep clone
```

... the end
