# stringset.js
A fast and robust stringset implementation that can hold any string items,
including `__proto__`, with minimal overhead compared to a plain object.
Works in node and browsers.

The API is created to be as close to the ES6 Set API as possible. Prefer
`ss.remove("key")` for deleting a key. ES6 Set uses `set.delete("key")`
instead and for that reason `ss['delete']("key")` is available as a
stringset alias as well. Never do `ss.delete("key")` unless you're
certain to be in the land of ES5 or later.



## Examples
Available in `examples.js`

```javascript
var StringSet = require("stringset");

var ss1 = new StringSet();
ss1.add("greeting");
ss1.add("check");
ss1.add("__proto__");
console.log(ss1.has("greeting")); // true
console.log(ss1.has("__proto__")); // true
ss1.remove("greeting");
console.log(ss1.items()); // [ 'check', '__proto__' ]
console.log(ss1.toString()); // {"check","__proto__"}

var ss2 = new StringSet(["one", "two"]);
console.log(ss2.isEmpty()); // false
console.log(ss2.size()); // 2

var ss3 = ss1.clone();
ss3.merge(ss2);
ss3.addMany(["a", "b"]);
console.log(ss3.toString()); // {"check","one","two","a","b","__proto__"}
```



## Installation

### Node
Install using npm

    npm install stringset

```javascript
var StringSet = require("stringset");
```

### Browser
Clone the repo and include it in a script tag

    git clone https://github.com/olov/stringset.git

```html
<script src="stringset/stringset.js"></script>
```
