# unique-stream

node.js through stream that emits a unique stream of objects based on criteria

[![build status](https://secure.travis-ci.org/eugeneware/unique-stream.png)](http://travis-ci.org/eugeneware/unique-stream)

## Installation

Install via npm:

```
$ npm install unique-stream
```

## Examples

### Dedupe a ReadStream based on JSON.stringify:

``` js
var unique = require('unique-stream')
  , Stream = require('stream');

// return a stream of 3 identical objects
function makeStreamOfObjects() {
  var s = new Stream;
  s.readable = true;
  var count = 3;
  for (var i = 0; i < 3; i++) {
    setImmediate(function () {
      s.emit('data', { name: 'Bob', number: 123 });
      --count && end();
    });
  }

  function end() {
    s.emit('end');
  }

  return s;
}

// Will only print out one object as the rest are dupes. (Uses JSON.stringify)
makeStreamOfObjects()
  .pipe(unique())
  .on('data', console.log);

```

### Dedupe a ReadStream based on an object property:

``` js
// Use name as the key field to dedupe on. Will only print one object
makeStreamOfObjects()
  .pipe(unique('name'))
  .on('data', console.log);
```

### Dedupe a ReadStream based on a custom function:

``` js
// Use a custom function to dedupe on. Use the 'number' field. Will only print one object.
makeStreamOfObjects()
  .pipe(function (data) {
    return data.number;
  })
  .on('data', console.log);
```

## Dedupe multiple streams

The reason I wrote this was to dedupe multiple object streams:

``` js
var aggregator = unique();

// Stream 1
makeStreamOfObjects()
  .pipe(aggregator);

// Stream 2
makeStreamOfObjects()
  .pipe(aggregator);

// Stream 3
makeStreamOfObjects()
  .pipe(aggregator);

aggregator.on('data', console.log);
```
