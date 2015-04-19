# stringmap.js
A fast and robust stringmap implementation that can hold any string keys,
including `__proto__`, with minimal overhead compared to a plain object.
Works in node and browsers.

The API is created to be as close to the ES6 Map API as possible. Prefer
`sm.remove("key")` for deleting a key. ES6 Map uses `map.delete("key")`
instead and for that reason `sm['delete']("key")` is available as a
stringmap alias as well. Never do `sm.delete("key")` unless you're
certain to be in the land of ES5 or later.



## Examples
Available in `examples.js`

```javascript
var StringMap = require("stringmap");

var sm1 = new StringMap();
sm1.set("greeting", "yoyoma");
sm1.set("check", true);
sm1.set("__proto__", -1);
console.log(sm1.has("greeting")); // true
console.log(sm1.get("__proto__")); // -1
sm1.remove("greeting");
console.log(sm1.keys()); // [ 'check', '__proto__' ]
console.log(sm1.values()); // [ true, -1 ]
console.log(sm1.items()); // [ [ 'check', true ], [ '__proto__', -1 ] ]
console.log(sm1.toString()); // {"check":true,"__proto__":-1}

var sm2 = new StringMap({
    one: 1,
    two: 2,
});
console.log(sm2.map(function(value, key) {
    return value * value;
})); // [ 1, 4 ]
sm2.forEach(function(value, key) {
    // ...
});
console.log(sm2.isEmpty()); // false
console.log(sm2.size()); // 2

var sm3 = sm1.clone();
sm3.merge(sm2);
sm3.setMany({
    a: {},
    b: [],
});
console.log(sm3.toString()); // {"check":true,"one":1,"two":2,"a":{},"b":[],"__proto__":-1}
```



## Installation

### Node
Install using npm

    npm install stringmap

```javascript
var StringMap = require("stringmap");
```

### Browser
Clone the repo and include it in a script tag

    git clone https://github.com/olov/stringmap.git

```html
<script src="stringmap/stringmap.js"></script>
```
