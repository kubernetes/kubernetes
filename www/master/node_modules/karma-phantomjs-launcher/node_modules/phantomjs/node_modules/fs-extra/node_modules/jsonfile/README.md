[![build status](https://secure.travis-ci.org/jprichardson/node-jsonfile.png)](http://travis-ci.org/jprichardson/node-jsonfile)

Node.js - jsonfile
================

Easily read/write JSON files. 


Why?
----

Writing `JSON.stringify()` and then `fs.writeFile()` and `JSON.parse()` with `fs.readFile()` enclosed in `try/catch` blocks became annoying.



Installation
------------

    npm install jsonfile --save



API
---

### readFile(filename, [options], callback)

```javascript
var jf = require('jsonfile')
var util = require('util')

var file = '/tmp/data.json'
jf.readFile(file, function(err, obj) {
  console.log(util.inspect(obj))
})
```


### readFileSync(filename, [options])

```javascript
var jf = require('jsonfile')
var util = require('util')

var file = '/tmp/data.json'

console.log(util.inspect(jf.readFileSync(file)))
```

**options**: `throws`. Set to `false` if you don't ever want this method to throw on invalid JSON. Will return `null` instead. Defaults to `true`. Others passed directly to `fs.readFileSync`. 


### writeFile(filename, [options], callback)

```javascript
var jf = require('jsonfile')

var file = '/tmp/data.json'
var obj = {name: 'JP'}

jf.writeFile(file, obj, function(err) {
  console.log(err)
})
```

### writeFileSync(filename, [options])

```javascript
var jf = require('jsonfile')

var file = '/tmp/data.json'
var obj = {name: 'JP'}

jf.writeFileSync(file, obj)
```


### spaces

Number of spaces to indent JSON files. 

**default:** 2

```
var jf = require('jsonfile')

jf.spaces = 4;

var file = '/tmp/data.json'
var obj = {name: 'JP'}

jf.writeFile(file, obj, function(err) { //json file has four space indenting now
  console.log(err)
})
```


Contributions
-------------

If you contribute to this library, please don't change the version numbers in your pull request.


### Contributors

(You can add your name, or I'll add it if you forget)

- [*] [JP Richardson](https://github.com/jprichardson)
- [2] [Sean O'Dell](https://github.com/seanodell)
- [1] [Federico Fissore](https://github.com/ffissore)
- [1] [Ivan McCarthy](https://github.com/imcrthy)
- [1] [Pablo Vallejo](https://github.com/PabloVallejo)
- [1] [Miroslav Bajtoš](https://github.com/bajtos)


License
-------

(MIT License)

Copyright 2012-2014, JP Richardson  <jprichardson@gmail.com>





